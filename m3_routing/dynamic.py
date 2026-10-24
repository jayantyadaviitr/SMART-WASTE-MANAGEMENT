"""Visit only zones at or above the fill threshold. Shortest tour, then back to depot."""

from __future__ import annotations

from m3_routing.metrics import haversine_km, path_length_km


def _dist_matrix_m(points: list[tuple[float, float]]) -> list[list[int]]:
    n = len(points)
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mat[i][j] = int(round(haversine_km(points[i], points[j]) * 1000))
    return mat


def _nn_two_opt(points: list[tuple[float, float]]) -> list[int]:
    """Nearest neighbour + 2-opt. Depot is index 0. Returns visit order including 0 at both ends."""
    n = len(points)
    unused = set(range(1, n))
    order = [0]
    while unused:
        last = order[-1]
        nxt = min(unused, key=lambda j: haversine_km(points[last], points[j]))
        unused.remove(nxt)
        order.append(nxt)
    order.append(0)

    def tour_km(ord_):
        return sum(haversine_km(points[ord_[i]], points[ord_[i + 1]]) for i in range(len(ord_) - 1))

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new = order[:i] + order[i:j][::-1] + order[j:]
                if tour_km(new) + 1e-9 < tour_km(order):
                    order = new
                    improved = True
    return order


def _ortools_order(points: list[tuple[float, float]]) -> list[int] | None:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception:
        return None
    n = len(points)
    if n <= 2:
        return [0, 1, 0] if n == 2 else [0, 0]
    dist = _dist_matrix_m(points)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cb(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return dist[a][b]

    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(2)
    sol = routing.SolveWithParameters(params)
    if not sol:
        return None
    idx = routing.Start(0)
    order = []
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    order.append(0)
    return order


def dynamic_tour(depot: dict, zones: list[dict], fill_threshold: float = 60.0) -> dict:
    keep = [z for z in zones if float(z["combined"]) >= fill_threshold]
    skipped = [z for z in zones if float(z["combined"]) < fill_threshold]
    if not keep:
        coords = [(depot["lat"], depot["lon"]), (depot["lat"], depot["lon"])]
        return {
            "name": "Dynamic loop (fill threshold)",
            "stops": [],
            "coords": coords,
            "km": 0.0,
            "solver": "none (no zone above threshold)",
            "skipped": skipped,
        }

    points = [(depot["lat"], depot["lon"])] + [(z["lat"], z["lon"]) for z in keep]
    solver = "ortools"
    order = _ortools_order(points)
    if order is None:
        order = _nn_two_opt(points)
        solver = "nearest-neighbour + 2-opt (OR-Tools unavailable)"

    stops = []
    coords = []
    for node in order:
        if node == 0:
            coords.append((depot["lat"], depot["lon"]))
        else:
            z = keep[node - 1]
            stops.append(z)
            coords.append((z["lat"], z["lon"]))
    # order includes depot at start and end; stops should not repeat depot
    # Rebuild stops in visit order without the depot entries
    visit_stops = []
    for node in order:
        if node == 0:
            continue
        visit_stops.append(keep[node - 1])
    km = path_length_km(coords)
    return {
        "name": "Dynamic loop (critical zones only)",
        "stops": visit_stops,
        "coords": coords,
        "km": km,
        "solver": solver,
        "skipped": skipped,
    }


def osrm_geometry(coords: list[tuple[float, float]], timeout: float = 6.0) -> tuple[list[tuple[float, float]], str]:
    """Try public OSRM for road shape. Fall back to the straight-line coords."""
    if len(coords) < 2:
        return coords, "straight-line"
    try:
        import requests

        loc = ";".join(f"{lon},{lat}" for lat, lon in coords)
        url = f"https://router.project-osrm.org/route/v1/driving/{loc}"
        r = requests.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=timeout)
        r.raise_for_status()
        geom = r.json()["routes"][0]["geometry"]["coordinates"]
        path = [(latlon[1], latlon[0]) for latlon in geom]
        return path, "OSRM roads"
    except Exception:
        return coords, "straight-line (OSRM unreachable)"
