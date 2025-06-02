import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import heapq
from collections import defaultdict
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random

# Sabit veri setleri
DRONES_DATA = [
    {"id": 1, "max_weight": 4.0, "battery": 12000, "speed": 8.0, "start_pos": (10, 10)},
    {"id": 2, "max_weight": 3.5, "battery": 10000, "speed": 10.0, "start_pos": (20, 30)},
    {"id": 3, "max_weight": 5.0, "battery": 15000, "speed": 7.0, "start_pos": (50, 50)},
    {"id": 4, "max_weight": 2.0, "battery": 8000, "speed": 12.0, "start_pos": (80, 20)},
    {"id": 5, "max_weight": 6.0, "battery": 20000, "speed": 5.0, "start_pos": (40, 70)}
]

DELIVERIES_DATA = [
    {"id": 1, "pos": (15, 25), "weight": 1.5, "priority": 3, "time_window": (0, 60)},
    {"id": 2, "pos": (30, 40), "weight": 2.0, "priority": 5, "time_window": (0, 30)},
    {"id": 3, "pos": (70, 80), "weight": 3.0, "priority": 2, "time_window": (20, 80)},
    {"id": 4, "pos": (90, 10), "weight": 1.0, "priority": 4, "time_window": (10, 40)},
    {"id": 5, "pos": (45, 60), "weight": 4.0, "priority": 1, "time_window": (30, 90)},
    {"id": 6, "pos": (25, 15), "weight": 2.5, "priority": 3, "time_window": (0, 50)},
    {"id": 7, "pos": (60, 30), "weight": 1.0, "priority": 5, "time_window": (5, 25)},
    {"id": 8, "pos": (85, 90), "weight": 3.5, "priority": 2, "time_window": (40, 100)},
    {"id": 9, "pos": (10, 80), "weight": 2.0, "priority": 4, "time_window": (15, 45)},
    {"id": 10, "pos": (95, 50), "weight": 1.5, "priority": 3, "time_window": (0, 60)},
    {"id": 11, "pos": (55, 20), "weight": 0.5, "priority": 5, "time_window": (0, 20)},
    {"id": 12, "pos": (35, 75), "weight": 2.0, "priority": 1, "time_window": (50, 120)},
    {"id": 13, "pos": (75, 40), "weight": 3.0, "priority": 3, "time_window": (10, 50)},
    {"id": 14, "pos": (20, 90), "weight": 1.5, "priority": 4, "time_window": (30, 70)},
    {"id": 15, "pos": (65, 65), "weight": 4.5, "priority": 2, "time_window": (25, 75)},
    {"id": 16, "pos": (40, 10), "weight": 2.0, "priority": 5, "time_window": (0, 30)},
    {"id": 17, "pos": (5, 50), "weight": 1.0, "priority": 3, "time_window": (15, 55)},
    {"id": 18, "pos": (50, 85), "weight": 3.0, "priority": 1, "time_window": (60, 100)},
    {"id": 19, "pos": (80, 70), "weight": 2.5, "priority": 4, "time_window": (20, 60)},
    {"id": 20, "pos": (30, 55), "weight": 1.5, "priority": 2, "time_window": (40, 80)}
]

NO_FLY_ZONES_DATA = [
    {
        "id": 1,
        "coordinates": [(40, 30), (60, 30), (60, 50), (40, 50)],
        "active_time": (0, 120)
    },
    {
        "id": 2,
        "coordinates": [(70, 10), (90, 10), (90, 30), (70, 30)],
        "active_time": (30, 90)
    },
    {
        "id": 3,
        "coordinates": [(10, 60), (30, 60), (30, 80), (10, 80)],
        "active_time": (0, 60)
    }
]


@dataclass
class Drone:
    id: int
    max_weight: float
    battery: int
    speed: float
    start_pos: Tuple[float, float]
    current_battery: float = 100.0
    current_pos: Tuple[float, float] = None

    def __post_init__(self):
        if self.current_pos is None:
            self.current_pos = self.start_pos


@dataclass
class Package:
    id: int
    pos: Tuple[float, float]
    weight: float
    priority: int
    time_window: Tuple[str, str]
    delivered: bool = False


@dataclass
class NoFlyZone:
    id: int
    coordinates: List[Tuple[float, float]]
    active_time: Tuple[str, str]

    def is_active(self, current_time: datetime) -> bool:
        start_time = datetime.strptime(self.active_time[0], "%H:%M").time()
        end_time = datetime.strptime(self.active_time[1], "%H:%M").time()
        return start_time <= current_time.time() <= end_time


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class DroneDeliveryOptimizer:
    def __init__(self, start_time=None):
        self.drones = []
        self.packages = []
        self.no_fly_zones = []

        # Başlangıç zamanını ayarla (debug mode için)
        if start_time:
            try:
                # Format: "HH:MM" veya "YYYY-MM-DD HH:MM"
                if len(start_time.split()) == 1:  # Sadece saat verilmişse
                    hour, minute = map(int, start_time.split(':'))
                    self.current_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:  # Tam tarih verilmişse
                    self.current_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
                print(f"🕐 Başlangıç zamanı: {self.current_time.strftime('%Y-%m-%d %H:%M')}")
            except ValueError:
                print(f"❌ Hatalı saat formatı: {start_time}")
                print("💡 Doğru format: 'HH:MM' veya 'YYYY-MM-DD HH:MM'")
                print("🔄 Varsayılan saat (09:00) kullanılacak...")
                self.current_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        else:
            # Varsayılan: bugün saat 09:00
            self.current_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
            print(f"🕘 Varsayılan başlangıç zamanı: {self.current_time.strftime('%Y-%m-%d %H:%M')}")

        self.load_data()

    def load_data(self):
        """Sabit veri setini yükle"""
        # Drone'ları yükle
        for d in DRONES_DATA:
            drone = Drone(
                id=d["id"] - 1,
                max_weight=d["max_weight"],
                battery=d["battery"],
                speed=d["speed"],
                start_pos=d["start_pos"]
            )
            self.drones.append(drone)

        # Paketleri yükle
        base_time = self.current_time
        for d in DELIVERIES_DATA:
            start_minutes = d["time_window"][0]
            end_minutes = d["time_window"][1]

            start_time = base_time + timedelta(minutes=start_minutes)
            end_time = base_time + timedelta(minutes=end_minutes)

            package = Package(
                id=d["id"] - 1,
                pos=d["pos"],
                weight=d["weight"],
                priority=d["priority"],
                time_window=(start_time.strftime("%H:%M"), end_time.strftime("%H:%M"))
            )
            self.packages.append(package)

        # No-fly zone'ları yükle
        for nfz in NO_FLY_ZONES_DATA:
            start_minutes = nfz["active_time"][0]
            end_minutes = nfz["active_time"][1]

            start_time = base_time + timedelta(minutes=start_minutes)
            end_time = base_time + timedelta(minutes=end_minutes)

            zone = NoFlyZone(
                id=nfz["id"] - 1,
                coordinates=nfz["coordinates"],
                active_time=(start_time.strftime("%H:%M"), end_time.strftime("%H:%M"))
            )
            self.no_fly_zones.append(zone)

    def calculate_energy_consumption(self, distance: float, weight: float) -> float:
        base_consumption = 0.01  # %/metre
        weight_factor = 1 + (weight / 10)
        return distance * base_consumption * weight_factor

    def is_zone_active_at_time(self, zone: NoFlyZone, check_time: datetime = None) -> bool:
        """Belirli bir zamanda zone'un aktif olup olmadığını kontrol et"""
        if check_time is None:
            check_time = self.current_time

        start_time = datetime.strptime(zone.active_time[0], "%H:%M").time()
        end_time = datetime.strptime(zone.active_time[1], "%H:%M").time()
        current_time = check_time.time()

        return start_time <= current_time <= end_time

    def get_active_no_fly_zones(self, check_time: datetime = None) -> List[NoFlyZone]:
        """Aktif no-fly zone'ları döndür"""
        if check_time is None:
            check_time = self.current_time

        active_zones = []
        for zone in self.no_fly_zones:
            if self.is_zone_active_at_time(zone, check_time):
                active_zones.append(zone)
        return active_zones

    def check_path_through_active_no_fly_zones(self, start: Tuple[float, float], end: Tuple[float, float],
                                               check_time: datetime = None) -> Tuple[bool, List[NoFlyZone]]:
        """Yolun aktif no-fly zone'lardan geçip geçmediğini kontrol et"""
        if check_time is None:
            check_time = self.current_time

        active_zones = self.get_active_no_fly_zones(check_time)
        violated_zones = []

        for zone in active_zones:
            # Yol üzerinde örnekleme yaparak kontrol et
            for i in range(20):  # Daha detaylı kontrol
                t = i / 19
                point = (start[0] + t * (end[0] - start[0]),
                         start[1] + t * (end[1] - start[1]))
                if point_in_polygon(point, zone.coordinates):
                    violated_zones.append(zone)
                    break

        return len(violated_zones) > 0, violated_zones

    def get_zone_waypoints(self, zone: NoFlyZone, start: Tuple[float, float], end: Tuple[float, float]) -> List[
        Tuple[float, float]]:
        """Zone'un köşe noktalarını ve ek waypoint'leri döndür"""
        waypoints = []

        # Zone köşeleri
        corners = zone.coordinates.copy()

        # Zone'un etrafında ek noktalar ekle (güvenlik mesafesi)
        safety_margin = 5.0

        # Zone'un merkezi
        center_x = sum(p[0] for p in corners) / len(corners)
        center_y = sum(p[1] for p in corners) / len(corners)

        # Her köşe için güvenlik mesafeli noktalar ekle
        for corner in corners:
            # Köşeden merkeze doğru vektör
            to_center_x = center_x - corner[0]
            to_center_y = center_y - corner[1]

            # Normalize et
            length = np.sqrt(to_center_x ** 2 + to_center_y ** 2)
            if length > 0:
                to_center_x /= length
                to_center_y /= length

            # Güvenlik mesafesi kadar dışarıya çık
            safe_point = (
                corner[0] - to_center_x * safety_margin,
                corner[1] - to_center_y * safety_margin
            )
            waypoints.append(safe_point)

        return waypoints

    def find_shortest_path_avoiding_zones(self, start: Tuple[float, float], end: Tuple[float, float],
                                          check_time: datetime = None) -> Tuple[List[Tuple[float, float]], float]:
        """Aktif no-fly zone'ları kaçınan en kısa yolu bul"""
        if check_time is None:
            check_time = self.current_time

        # Önce direkt yolu kontrol et
        has_violation, _ = self.check_path_through_active_no_fly_zones(start, end, check_time)
        if not has_violation:
            distance = euclidean_distance(start, end)
            return [start, end], distance

        # Aktif zone'ları al
        active_zones = self.get_active_no_fly_zones(check_time)

        # Tüm waypoint'leri topla
        all_waypoints = [start, end]
        for zone in active_zones:
            waypoints = self.get_zone_waypoints(zone, start, end)
            all_waypoints.extend(waypoints)

        # Dijkstra algoritması için graf oluştur
        n = len(all_waypoints)
        distances = {}

        # Her nokta çifti arasındaki mesafeyi hesapla
        for i in range(n):
            for j in range(i + 1, n):
                point1 = all_waypoints[i]
                point2 = all_waypoints[j]

                # Bu iki nokta arasında aktif zone geçişi var mı?
                has_violation, _ = self.check_path_through_active_no_fly_zones(point1, point2, check_time)

                if not has_violation:
                    dist = euclidean_distance(point1, point2)
                    distances[(i, j)] = dist
                    distances[(j, i)] = dist

        # Dijkstra algoritması
        start_idx = 0  # start noktası
        end_idx = 1  # end noktası

        dist = [float('inf')] * n
        dist[start_idx] = 0
        prev = [-1] * n
        visited = [False] * n

        for _ in range(n):
            # En küçük mesafeli ziyaret edilmemiş nokta
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or dist[v] < dist[u]):
                    u = v

            if u == -1 or dist[u] == float('inf'):
                break

            visited[u] = True

            # Komşuları güncelle
            for v in range(n):
                if not visited[v] and (u, v) in distances:
                    alt = dist[u] + distances[(u, v)]
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u

        # Yolu yeniden oluştur
        if dist[end_idx] == float('inf'):
            # Yol bulunamadı, direkt yolu döndür (son çare)
            distance = euclidean_distance(start, end)
            return [start, end], distance

        path = []
        current = end_idx
        while current != -1:
            path.append(all_waypoints[current])
            current = prev[current]

        path.reverse()
        total_distance = dist[end_idx]

        return path, total_distance

    def check_path_through_no_fly_zone(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Eski fonksiyonu yeni sisteme uyarla"""
        has_violation, _ = self.check_path_through_active_no_fly_zones(start, end)
        return has_violation

    def find_alternative_path(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Eski fonksiyonu yeni sisteme uyarla"""
        path, _ = self.find_shortest_path_avoiding_zones(start, end)
        return path

    def calculate_travel_time(self, distance: float, speed: float) -> float:
        """Seyahat süresini saniye cinsinden hesapla"""
        return distance / speed if speed > 0 else 0

    def is_package_in_time_window(self, package: Package, arrival_time: datetime) -> bool:
        """Paketin zaman penceresi içinde teslim edilip edilemeyeceğini kontrol et"""
        start_time = datetime.strptime(package.time_window[0], "%H:%M").replace(
            year=arrival_time.year, month=arrival_time.month, day=arrival_time.day
        )
        end_time = datetime.strptime(package.time_window[1], "%H:%M").replace(
            year=arrival_time.year, month=arrival_time.month, day=arrival_time.day
        )

        # Eğer end_time start_time'dan küçükse, ertesi gün demektir
        if end_time < start_time:
            end_time += timedelta(days=1)

        return start_time <= arrival_time <= end_time

    def solve_with_greedy(self) -> Dict[int, List[Package]]:
        """Dinamik zamanlı Greedy algoritma ile çözüm"""
        assignments = defaultdict(list)

        # Her drone'un mevcut zamanını takip et
        drone_times = {drone.id: self.current_time for drone in self.drones}

        # Paketleri öncelik ve mesafeye göre sırala
        def package_score(package):
            min_distance = min(euclidean_distance(drone.current_pos, package.pos) for drone in self.drones)
            return package.priority * 100 - min_distance

        sorted_packages = sorted(self.packages, key=package_score, reverse=True)

        for package in sorted_packages:
            best_drone = None
            best_cost = float('inf')
            best_arrival_time = None

            for drone in self.drones:
                # Ağırlık kontrolü
                if package.weight > drone.max_weight:
                    continue

                # Drone'un mevcut zamanı
                drone_current_time = drone_times[drone.id]

                # En kısa güvenli yolu bul (drone'un mevcut zamanında)
                path, total_distance = self.find_shortest_path_avoiding_zones(
                    drone.current_pos, package.pos, drone_current_time
                )

                # Seyahat süresi hesapla
                travel_time = self.calculate_travel_time(total_distance, drone.speed)
                arrival_time = drone_current_time + timedelta(seconds=travel_time)

                # Zaman penceresi kontrolü
                if not self.is_package_in_time_window(package, arrival_time):
                    continue  # Zaman penceresi dışında, bu drone uygun değil

                # Dönüş yolu ve süresi
                return_path, return_distance = self.find_shortest_path_avoiding_zones(
                    package.pos, drone.start_pos, arrival_time
                )
                return_travel_time = self.calculate_travel_time(return_distance, drone.speed)

                total_distance = total_distance + return_distance
                total_travel_time = travel_time + return_travel_time

                # Toplam enerji tüketimi
                energy_to_package = self.calculate_energy_consumption(total_distance - return_distance, package.weight)
                energy_return = self.calculate_energy_consumption(return_distance, 0)
                total_energy = energy_to_package + energy_return

                # Enerji kontrolü
                if total_energy > drone.current_battery:
                    continue

                # No-fly zone ihlal cezası (dinamik zaman kontrolü)
                violation_penalty = 0
                has_violation_to, _ = self.check_path_through_active_no_fly_zones(
                    drone.current_pos, package.pos, drone_current_time
                )
                has_violation_return, _ = self.check_path_through_active_no_fly_zones(
                    package.pos, drone.start_pos, arrival_time
                )

                if has_violation_to:
                    violation_penalty += 10000
                if has_violation_return:
                    violation_penalty += 10000

                # Zaman cezası ekle (geç teslimat)
                time_penalty = 0
                package_end_time = datetime.strptime(package.time_window[1], "%H:%M").replace(
                    year=arrival_time.year, month=arrival_time.month, day=arrival_time.day
                )
                if arrival_time > package_end_time:
                    delay_minutes = (arrival_time - package_end_time).total_seconds() / 60
                    time_penalty = delay_minutes * 50  # Dakika başına 50 puan ceza

                # Maliyet hesapla
                cost = (total_distance * package.weight +
                        (package.priority * 100) +
                        violation_penalty +
                        time_penalty +
                        total_travel_time * 0.1)  # Zaman faktörü

                if cost < best_cost:
                    best_cost = cost
                    best_drone = drone
                    best_arrival_time = arrival_time + timedelta(seconds=return_travel_time)

            if best_drone:
                assignments[best_drone.id].append(package)

                # Drone'u güncelle
                path, distance = self.find_shortest_path_avoiding_zones(
                    best_drone.current_pos, package.pos, drone_times[best_drone.id]
                )
                return_path, return_distance = self.find_shortest_path_avoiding_zones(
                    package.pos, best_drone.start_pos, best_arrival_time - timedelta(seconds=return_travel_time)
                )

                # Enerji tüketimi güncelle
                energy_to_package = self.calculate_energy_consumption(distance, package.weight)
                energy_return = self.calculate_energy_consumption(return_distance, 0)
                best_drone.current_battery -= (energy_to_package + energy_return)

                # Zamanı güncelle
                drone_times[best_drone.id] = best_arrival_time

                # Drone başlangıç pozisyonuna döner
                best_drone.current_pos = best_drone.start_pos

        return assignments

    def solve_with_genetic(self, population_size=30, generations=50):
        """Dengeli genetik algoritma - Dijkstra korundu"""

        def create_individual():
            individual = []
            for package in self.packages:
                valid_drones = [d.id for d in self.drones if package.weight <= d.max_weight]
                if valid_drones:
                    # Yakın drone'ları tercih et (basit versiyon)
                    drone_distances = []
                    for drone_id in valid_drones:
                        drone = self.drones[drone_id]
                        distance = euclidean_distance(drone.start_pos, package.pos)
                        drone_distances.append((drone_id, distance))

                    # En yakın 2 drone arasından seç (3'ten 2'ye düşürüldü)
                    drone_distances.sort(key=lambda x: x[1])
                    top_drones = drone_distances[:min(2, len(drone_distances))]
                    drone_id = random.choice(top_drones)[0]

                    individual.append((package.id, drone_id))
                else:
                    individual.append((package.id, -1))
            return individual

        def fitness(individual):
            delivered = 0
            total_energy = 0
            no_fly_zone_violations = 0

            drone_routes = defaultdict(list)
            for package_id, drone_id in individual:
                if drone_id != -1:
                    drone_routes[drone_id].append(package_id)

            # Her drone'un başlangıç zamanı
            drone_times = {drone_id: self.current_time for drone_id in range(len(self.drones))}

            for drone_id, package_ids in drone_routes.items():
                drone = self.drones[drone_id]
                current_pos = drone.start_pos
                current_time = drone_times[drone_id]

                for package_id in package_ids:
                    package = self.packages[package_id]

                    # Dijkstra ile en kısa güvenli yolu kullan
                    path, distance = self.find_shortest_path_avoiding_zones(current_pos, package.pos, current_time)

                    # Seyahat süresi hesapla
                    travel_time = self.calculate_travel_time(distance, drone.speed)
                    arrival_time = current_time + timedelta(seconds=travel_time)

                    # Zaman penceresi kontrolü - sadece teslim sayısını etkiler
                    if self.is_package_in_time_window(package, arrival_time):
                        delivered += 1

                    # Enerji tüketimi hesapla
                    energy = self.calculate_energy_consumption(distance, package.weight)
                    total_energy += energy

                    # No-fly zone ihlali kontrol et (dinamik zaman)
                    has_violation, _ = self.check_path_through_active_no_fly_zones(current_pos, package.pos,
                                                                                   current_time)
                    if has_violation:
                        no_fly_zone_violations += 1

                    # Pozisyon ve zamanı güncelle
                    current_pos = package.pos
                    current_time = arrival_time

                # Depoya dönüş - Dijkstra ile
                return_path, return_distance = self.find_shortest_path_avoiding_zones(current_pos, drone.start_pos,
                                                                                      current_time)
                return_travel_time = self.calculate_travel_time(return_distance, drone.speed)
                return_arrival_time = current_time + timedelta(seconds=return_travel_time)

                return_energy = self.calculate_energy_consumption(return_distance, 0)
                total_energy += return_energy

                # Dönüş yolunda no-fly zone ihlali kontrol et
                has_violation, _ = self.check_path_through_active_no_fly_zones(current_pos, drone.start_pos,
                                                                               current_time)
                if has_violation:
                    no_fly_zone_violations += 1

                # Drone'un son zamanını güncelle
                drone_times[drone_id] = return_arrival_time

            # ORİJİNAL FITNESS FORMÜLÜ: teslimat sayısı * 50 - toplam enerji * 0.1 - ihlal sayısı * 1000
            return delivered * 50 - total_energy * 0.1 - no_fly_zone_violations * 1000

        def smart_crossover(parent1, parent2):
            """Basitleştirilmiş akıllı çaprazlama"""
            child = []
            for i in range(len(parent1)):
                package_id = parent1[i][0]
                package = self.packages[package_id]

                parent1_drone = parent1[i][1]
                parent2_drone = parent2[i][1]

                # Basit seçim: her iki drone da geçerliyse rastgele seç
                if parent1_drone != -1 and parent2_drone != -1:
                    chosen_drone = random.choice([parent1_drone, parent2_drone])
                elif parent1_drone != -1:
                    chosen_drone = parent1_drone
                elif parent2_drone != -1:
                    chosen_drone = parent2_drone
                else:
                    chosen_drone = -1

                child.append((package_id, chosen_drone))

            return child

        # Başlangıç popülasyonu
        population = [create_individual() for _ in range(population_size)]

        for generation in range(generations):
            # Fitness hesapla
            fitness_scores = [(ind, fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)

            # Basit elitizm: En iyi 2'yi koru
            new_population = [fitness_scores[0][0], fitness_scores[1][0]]

            # Yeni nesil oluştur
            while len(new_population) < population_size:
                # Basit seçim
                parent1 = random.choice(population[:population_size // 2])
                parent2 = random.choice(population[:population_size // 2])

                # Akıllı çaprazlama (basitleştirilmiş)
                child = smart_crossover(parent1, parent2)

                # Basit mutasyon
                if random.random() < 0.1:
                    idx = random.randint(0, len(child) - 1)
                    package_id, _ = child[idx]
                    package = self.packages[package_id]
                    valid_drones = [d.id for d in self.drones if package.weight <= d.max_weight]
                    if valid_drones:
                        child[idx] = (package_id, random.choice(valid_drones))

                new_population.append(child)

            population = new_population

            if generation % 10 == 0:
                print(f"  Generation {generation}: Best fitness = {fitness_scores[0][1]:.2f}")

        # En iyi çözümü döndür
        best_individual = max(population, key=fitness)
        assignments = defaultdict(list)

        for package_id, drone_id in best_individual:
            if drone_id != -1:
                assignments[drone_id].append(self.packages[package_id])

        return assignments

    def visualize_solution(self, assignments: Dict[int, List[Package]], title: str = "Drone Teslimat Rotaları"):
        plt.figure(figsize=(14, 10))

        # Harita ayarları
        plt.xlim(-5, 105)
        plt.ylim(-5, 105)
        plt.grid(True, alpha=0.3)

        # No-fly zone'ları çiz
        for zone in self.no_fly_zones:
            polygon = Polygon(zone.coordinates, alpha=0.3, color='red', edgecolor='darkred', linewidth=2)
            plt.gca().add_patch(polygon)
            center = np.mean(zone.coordinates, axis=0)
            plt.text(center[0], center[1], f'NFZ-{zone.id + 1}\n{zone.active_time[0]}-{zone.active_time[1]}',
                     ha='center', va='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Tüm paketleri göster
        all_assigned_packages = set()
        for packages in assignments.values():
            for package in packages:
                all_assigned_packages.add(package.id)

        for package in self.packages:
            if package.id not in all_assigned_packages:
                plt.scatter(package.pos[0], package.pos[1], s=80, c='gray', marker='o', alpha=0.5)
                plt.text(package.pos[0], package.pos[1] + 1.5, f'P{package.id + 1}',
                         fontsize=7, ha='center', color='gray')

        # Drone başlangıç noktaları
        for drone in self.drones:
            plt.scatter(drone.start_pos[0], drone.start_pos[1], s=200, c='green', marker='s',
                        edgecolor='darkgreen', linewidth=2)
            plt.text(drone.start_pos[0], drone.start_pos[1], f'D{drone.id + 1}',
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')

        # Rotaları çiz
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.drones)))
        total_distance = 0
        total_delivered = 0

        def paths_are_same(path1, path2):
            """İki yolun aynı olup olmadığını kontrol et"""
            if len(path1) != len(path2):
                return False
            # Ters yönü de kontrol et
            return path1 == path2 or path1 == path2[::-1]

        def draw_offset_path(path, color, linewidth, alpha, linestyle, offset_distance=1.5):
            """Paralel çizgi çizmek için offset hesapla"""
            for j in range(len(path) - 1):
                x1, y1 = path[j]
                x2, y2 = path[j + 1]

                # Perpendicular vektör hesapla
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx ** 2 + dy ** 2)

                if length > 0:
                    # Normalize ve perpendicular
                    unit_dx = dx / length
                    unit_dy = dy / length
                    perp_x = -unit_dy * offset_distance
                    perp_y = unit_dx * offset_distance

                    # Offset koordinatları
                    offset_x1 = x1 + perp_x
                    offset_y1 = y1 + perp_y
                    offset_x2 = x2 + perp_x
                    offset_y2 = y2 + perp_y

                    plt.plot([offset_x1, offset_x2], [offset_y1, offset_y2],
                             c=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
                else:
                    plt.plot([x1, x2], [y1, y2],
                             c=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)

        for drone_id, packages in assignments.items():
            if packages:
                drone = self.drones[drone_id]
                color = colors[drone_id]
                current_pos = drone.start_pos
                drone_distance = 0

                for i, package in enumerate(packages):
                    # Paket noktası
                    plt.scatter(package.pos[0], package.pos[1], s=120, c=color, marker='o',
                                edgecolor='black', linewidth=1)
                    plt.text(package.pos[0], package.pos[1] + 1.5,
                             f'P{package.id + 1}\n{package.weight:.1f}kg\nPri:{package.priority}',
                             fontsize=8, ha='center', va='bottom')

                    # Gidiş rotası
                    delivery_path = self.find_alternative_path(current_pos, package.pos)

                    # Dönüş rotası
                    return_path = self.find_alternative_path(package.pos, drone.start_pos)

                    # Gidiş ve dönüş rotaları aynı mı kontrol et
                    same_path = paths_are_same(delivery_path, return_path)

                    if same_path:
                        # 🔄 AYNI ROTA: Tek çizgi, çift yönlü ok
                        for j in range(len(delivery_path) - 1):
                            plt.plot([delivery_path[j][0], delivery_path[j + 1][0]],
                                     [delivery_path[j][1], delivery_path[j + 1][1]],
                                     c=color, linewidth=3, alpha=0.8)

                        # Çift yönlü ok ekle (ortada)
                        mid_idx = len(delivery_path) // 2
                        if mid_idx < len(delivery_path) - 1:
                            mid_point = delivery_path[mid_idx]
                            next_point = delivery_path[mid_idx + 1]

                            # Ok yönü
                            dx = next_point[0] - mid_point[0]
                            dy = next_point[1] - mid_point[1]

                            plt.annotate('', xy=(mid_point[0] + dx * 0.3, mid_point[1] + dy * 0.3),
                                         xytext=(mid_point[0] - dx * 0.3, mid_point[1] - dy * 0.3),
                                         arrowprops=dict(arrowstyle='<->', color=color, lw=2))

                        # "Aynı Rota" etiketi
                        path_mid = delivery_path[len(delivery_path) // 2]
                        plt.text(path_mid[0], path_mid[1] - 2, f'↔ #{i + 1}',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9),
                                 ha='center', va='center', fontsize=8, color='white', fontweight='bold')

                    else:
                        # 🔀 FARKLI ROTALAR: Parallel çizgiler

                        # Gidiş rotası (kalın çizgi)
                        for j in range(len(delivery_path) - 1):
                            plt.plot([delivery_path[j][0], delivery_path[j + 1][0]],
                                     [delivery_path[j][1], delivery_path[j + 1][1]],
                                     c=color, linewidth=2.5, alpha=0.8, label=f'Gidiş-{i + 1}' if j == 0 else "")

                        # Dönüş rotası (kesikli, offset)
                        draw_offset_path(return_path, color, linewidth=2, alpha=0.6, linestyle='--')

                        # Gidiş ok işareti
                        delivery_mid = delivery_path[len(delivery_path) // 2]
                        if len(delivery_path) > 1:
                            next_point = delivery_path[len(delivery_path) // 2 + 1] if len(
                                delivery_path) // 2 + 1 < len(delivery_path) else delivery_path[-1]
                            dx = next_point[0] - delivery_mid[0]
                            dy = next_point[1] - delivery_mid[1]
                            plt.annotate('', xy=(delivery_mid[0] + dx * 0.2, delivery_mid[1] + dy * 0.2),
                                         xytext=delivery_mid,
                                         arrowprops=dict(arrowstyle='->', color=color, lw=2))

                        # Dönüş ok işareti (offset)
                        return_mid = return_path[len(return_path) // 2]
                        if len(return_path) > 1:
                            next_point = return_path[len(return_path) // 2 + 1] if len(return_path) // 2 + 1 < len(
                                return_path) else return_path[-1]
                            dx = next_point[0] - return_mid[0]
                            dy = next_point[1] - return_mid[1]

                            # Offset hesapla
                            length = np.sqrt(dx ** 2 + dy ** 2)
                            if length > 0:
                                unit_dx = dx / length
                                unit_dy = dy / length
                                perp_x = -unit_dy * 1.5
                                perp_y = unit_dx * 1.5

                                offset_mid = (return_mid[0] + perp_x, return_mid[1] + perp_y)
                                offset_next = (return_mid[0] + dx * 0.2 + perp_x, return_mid[1] + dy * 0.2 + perp_y)

                                plt.annotate('', xy=offset_next, xytext=offset_mid,
                                             arrowprops=dict(arrowstyle='->', color=color, lw=2, linestyle='--'))

                        # Rota numaraları
                        plt.text(delivery_mid[0], delivery_mid[1] + 1, f'G{i + 1}',
                                 bbox=dict(boxstyle="circle,pad=0.3", facecolor=color, alpha=0.8),
                                 ha='center', va='center', fontsize=8, color='white')

                        plt.text(return_mid[0] + 1.5, return_mid[1] + 1.5, f'D{i + 1}',
                                 bbox=dict(boxstyle="circle,pad=0.3", facecolor=color, alpha=0.6),
                                 ha='center', va='center', fontsize=8, color='white')

                    # Mesafe hesaplama
                    delivery_distance = sum(euclidean_distance(delivery_path[j], delivery_path[j + 1]) for j in
                                            range(len(delivery_path) - 1))
                    return_distance = sum(
                        euclidean_distance(return_path[j], return_path[j + 1]) for j in range(len(return_path) - 1))
                    drone_distance += delivery_distance + return_distance

                    current_pos = package.pos
                    total_delivered += 1

                total_distance += drone_distance

                # Legend bilgisi
                route_info = f'Drone {drone_id + 1}: {len(packages)} paket, {drone_distance:.1f}m'
                plt.plot([], [], c=color, linewidth=3, label=route_info)

        plt.title(f'{title}\n{total_delivered}/{len(self.packages)} paket teslim edildi, '
                  f'Toplam mesafe: {total_distance:.1f}m\n'
                  f'↔ Aynı rota (çift yönlü) | → Gidiş | ⇢ Dönüş (kesikli)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('X Koordinatı (metre)')
        plt.ylabel('Y Koordinatı (metre)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def print_statistics(self, assignments: Dict[int, List[Package]], algorithm_name: str, exec_time: float):
        """İstatistikleri yazdır - dinamik zaman destekli"""
        print(f"\n{algorithm_name} Sonuçları:")
        print(f"Çalışma süresi: {exec_time:.3f} saniye")

        total_delivered = 0
        total_distance = 0
        total_energy = 0
        total_time = 0
        time_violations = 0

        # Her drone'un zamanını takip et
        drone_times = {drone.id: self.current_time for drone in self.drones}

        for drone_id, packages in assignments.items():
            if packages:
                drone = self.drones[drone_id]
                drone_distance = 0
                drone_energy = 0
                drone_time = 0
                current_pos = drone.start_pos
                current_time = drone_times[drone_id]

                print(f"\nDrone {drone_id + 1}:")
                print(f"  Atanan paketler: {[p.id + 1 for p in packages]}")
                print(f"  Toplam ağırlık: {sum(p.weight for p in packages):.2f} kg")

                for i, package in enumerate(packages):
                    # Yol ve süre hesapla
                    path, distance = self.find_shortest_path_avoiding_zones(current_pos, package.pos, current_time)
                    travel_time = self.calculate_travel_time(distance, drone.speed)
                    arrival_time = current_time + timedelta(seconds=travel_time)

                    # Zaman penceresi kontrolü
                    in_time_window = self.is_package_in_time_window(package, arrival_time)
                    if not in_time_window:
                        time_violations += 1

                    energy = self.calculate_energy_consumption(distance, package.weight)
                    drone_distance += distance
                    drone_energy += energy
                    drone_time += travel_time

                    status = "✅ Zamanında" if in_time_window else "❌ Zaman aşımı"
                    print(f"    Paket {package.id + 1}: {distance:.1f}m, {travel_time / 60:.1f}dk, "
                          f"Varış: {arrival_time.strftime('%H:%M:%S')} {status}")

                    current_pos = package.pos
                    current_time = arrival_time

                # Depoya dönüş
                return_path, return_distance = self.find_shortest_path_avoiding_zones(current_pos, drone.start_pos,
                                                                                      current_time)
                return_time = self.calculate_travel_time(return_distance, drone.speed)
                final_time = current_time + timedelta(seconds=return_time)

                drone_distance += return_distance
                drone_energy += return_distance * 0.01
                drone_time += return_time

                print(f"  Dönüş: {return_distance:.1f}m, {return_time / 60:.1f}dk")
                print(f"  Toplam mesafe: {drone_distance:.2f} m")
                print(f"  Toplam süre: {drone_time / 60:.1f} dakika")
                print(f"  Bitiş zamanı: {final_time.strftime('%H:%M:%S')}")
                print(f"  Enerji tüketimi: {drone_energy:.2f}%")

                total_distance += drone_distance
                total_energy += drone_energy
                total_time += drone_time
                total_delivered += len(packages)

                # Drone zamanını güncelle
                drone_times[drone_id] = final_time

        # En geç bitiş zamanı
        latest_finish = max(drone_times.values()) if drone_times else self.current_time
        total_operation_time = (latest_finish - self.current_time).total_seconds() / 60

        print(f"\nToplam İstatistikler:")
        print(f"  Başlangıç zamanı: {self.current_time.strftime('%H:%M:%S')}")
        print(f"  Bitiş zamanı: {latest_finish.strftime('%H:%M:%S')}")
        print(f"  Toplam operasyon süresi: {total_operation_time:.1f} dakika")
        print(
            f"  Teslim edilen paket: {total_delivered}/{len(self.packages)} ({total_delivered / len(self.packages) * 100:.1f}%)")
        print(f"  Zaman aşımı olan paket: {time_violations}")
        print(f"  Teslim edilmeyen paket: {len(self.packages) - total_delivered}")
        print(f"  Toplam mesafe: {total_distance:.2f} m")
        print(f"  Ortalama enerji tüketimi: {total_energy / len(self.drones):.2f}%")
        print(f"  Ortalama drone kullanım süresi: {total_time / len(self.drones) / 60:.1f} dakika")


def main():
    print("=" * 60)
    print("DRONE TESLİMAT ROTA OPTİMİZASYONU")
    print("=" * 60)

    # Debug mode için saat girişi
    print("\n🕐 BAŞLANGIÇ SAATİ AYARI (Debug Mode)")
    print("-" * 40)
    print("Formatlar:")
    print("  • Sadece saat: '14:30' (bugün saat 14:30)")
    print("  • Tam tarih: '2025-06-01 14:30'")
    print("  • Boş bırak: Varsayılan (09:00)")

    start_time_input = input("\nBaşlangıç saati girin (Enter = varsayılan): ").strip()

    if start_time_input:
        optimizer = DroneDeliveryOptimizer(start_time=start_time_input)
    else:
        optimizer = DroneDeliveryOptimizer()

    print(f"\nYüklenen Veri:")
    print(f"  - {len(optimizer.drones)} Drone")
    print(f"  - {len(optimizer.packages)} Paket")
    print(f"  - {len(optimizer.no_fly_zones)} No-Fly Zone")

    # No-fly zone'ların aktif durumunu göster
    print(f"\n📍 NO-FLY ZONE DURUMLARI ({optimizer.current_time.strftime('%H:%M')} saatinde):")
    for i, zone in enumerate(optimizer.no_fly_zones):
        is_active = optimizer.is_zone_active_at_time(zone)
        status = "🔴 AKTİF" if is_active else "🟢 PASİF"
        print(f"  Zone {i + 1}: {zone.active_time[0]}-{zone.active_time[1]} {status}")

    active_zones = optimizer.get_active_no_fly_zones()
    print(f"\n⚠️  Aktif No-Fly Zone Sayısı: {len(active_zones)}")
    if active_zones:
        print("   Drone'lar bu zone'ları kaçınacak ve alternatif rotalar kullanacak!")

    # Greedy Algoritma
    print("\n" + "-" * 60)
    print("GREEDY ALGORİTMA")
    print("-" * 60)

    start_time = time.time()
    greedy_assignments = optimizer.solve_with_greedy()
    greedy_time = time.time() - start_time

    optimizer.print_statistics(greedy_assignments, "Greedy Algoritma", greedy_time)
    optimizer.visualize_solution(greedy_assignments, "Greedy Algoritma - Drone Teslimat Rotaları")

    # Drone'ları resetle
    for drone in optimizer.drones:
        drone.current_pos = drone.start_pos
        drone.current_battery = 100.0

    # Genetik Algoritma
    print("\n" + "-" * 60)
    print("GENETİK ALGORİTMA")
    print("-" * 60)

    start_time = time.time()
    genetic_assignments = optimizer.solve_with_genetic()
    genetic_time = time.time() - start_time

    optimizer.print_statistics(genetic_assignments, "Genetik Algoritma", genetic_time)
    optimizer.visualize_solution(genetic_assignments, "Genetik Algoritma - Drone Teslimat Rotaları")

    # Karşılaştırma
    print("\n" + "=" * 60)
    print("ALGORİTMA KARŞILAŞTIRMASI")
    print("=" * 60)

    greedy_delivered = sum(len(pkgs) for pkgs in greedy_assignments.values())
    genetic_delivered = sum(len(pkgs) for pkgs in genetic_assignments.values())

    print(f"\n{'Metrik':<30} {'Greedy':>15} {'Genetik':>15}")
    print("-" * 60)
    print(f"{'Teslim edilen paket':<30} {greedy_delivered:>15} {genetic_delivered:>15}")
    print(
        f"{'Teslimat yüzdesi (%)':<30} {greedy_delivered / len(optimizer.packages) * 100:>14.1f}% {genetic_delivered / len(optimizer.packages) * 100:>14.1f}%")
    print(f"{'Çalışma süresi (s)':<30} {greedy_time:>15.3f} {genetic_time:>15.3f}")

    print("\n" + "=" * 60)
    print("Program tamamlandı!")


if __name__ == "__main__":
    main()