"""
Compute the graph diameter of the facet dual graph.

Usage:
    python compute_mesh_diameter.py <input_file> <output_file>

Example:
    python compute_mesh_diameter.py it.txt output.txt

In the dual graph:
- Each triangular facet is a node (14050 nodes)
- Two facet-nodes are connected if they share an edge (i.e., two vertices) in the original mesh

We use BFS from multiple source nodes to estimate the diameter (the longest shortest path).
We also compute the eccentricity distribution to understand the graph's structure.
"""

import sys
import argparse
from collections import defaultdict, deque
import time
import random

def load_facets(filepath):
    """Load facets from file. Each row = 3 vertex indices."""
    facets = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 3:
                facets.append(tuple(int(x) for x in parts))
    return facets

def build_dual_graph(facets):
    """
    Build the dual graph adjacency list.
    Two facets are adjacent if they share exactly 2 vertices (i.e., an edge).
    """
    # Map each mesh edge (pair of vertices) to the facets that contain it
    edge_to_facets = defaultdict(list)
    for fi, (v0, v1, v2) in enumerate(facets):
        # Each triangle has 3 edges
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v0, v2))),
        ]
        for e in edges:
            edge_to_facets[e].append(fi)
    
    # Build adjacency list for the dual graph
    adj = defaultdict(set)
    for e, flist in edge_to_facets.items():
        for i in range(len(flist)):
            for j in range(i + 1, len(flist)):
                adj[flist[i]].add(flist[j])
                adj[flist[j]].add(flist[i])
    
    return adj

def bfs_eccentricity(adj, source, n):
    """BFS from source, return eccentricity (max distance) and distance array."""
    dist = [-1] * n
    dist[source] = 0
    queue = deque([source])
    max_dist = 0
    farthest = source
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                if dist[v] > max_dist:
                    max_dist = dist[v]
                    farthest = v
                queue.append(v)
    return max_dist, farthest, dist

def main():
    parser = argparse.ArgumentParser(description="Compute the graph diameter of the facet dual graph.")
    parser.add_argument("input_file", help="Path to the input text file containing facets")
    parser.add_argument("output_file", help="Path to the output text file to save results")
    args = parser.parse_args()
    
    # Redirect standard output to the output file using utf-8 encoding
    sys.stdout = open(args.output_file, 'w', encoding='utf-8')
    
    filepath = args.input_file
    
    print(f"Loading facets from {filepath}...")
    facets = load_facets(filepath)
    n = len(facets)
    print(f"  Loaded {n} facets")
    
    # Print vertex range
    all_verts = set()
    for v0, v1, v2 in facets:
        all_verts.update([v0, v1, v2])
    print(f"  Unique vertices: {len(all_verts)}")
    print(f"  Vertex range: {min(all_verts)} to {max(all_verts)}")
    
    print("\nBuilding dual graph...")
    t0 = time.time()
    adj = build_dual_graph(facets)
    t1 = time.time()
    print(f"  Built in {t1-t0:.2f}s")
    
    # Check connectivity and degree stats
    connected_nodes = len(adj)
    degrees = [len(adj[i]) for i in range(n)]
    print(f"  Nodes with neighbors: {connected_nodes}/{n}")
    print(f"  Degree stats: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/n:.1f}")
    
    # Estimate diameter using double-sweep BFS from multiple sources
    print("\nEstimating graph diameter...")
    
    # Strategy: BFS from a random node, then BFS from the farthest node found.
    # Repeat from several starting points.
    
    diameter_estimate = 0
    
    # First, try a few random seeds
    random.seed(42)
    sources = random.sample(range(n), min(10, n))
    
    for i, src in enumerate(sources):
        ecc1, far1, _ = bfs_eccentricity(adj, src, n)
        ecc2, far2, _ = bfs_eccentricity(adj, far1, n)
        local_diam = max(ecc1, ecc2)
        if local_diam > diameter_estimate:
            diameter_estimate = local_diam
            best_pair = (far1, far2) if ecc2 >= ecc1 else (src, far1)
        print(f"  Seed {i+1}/{len(sources)}: src={src}, sweep1_ecc={ecc1}, sweep2_ecc={ecc2}")
    
    # Do a few more sweeps from the endpoints of the best pair found
    for src in best_pair:
        ecc, far, _ = bfs_eccentricity(adj, src, n)
        if ecc > diameter_estimate:
            diameter_estimate = ecc
        # One more sweep
        ecc2, _, _ = bfs_eccentricity(adj, far, n)
        if ecc2 > diameter_estimate:
            diameter_estimate = ecc2
    
    print(f"\n  ESTIMATED GRAPH DIAMETER = {diameter_estimate}")
    
    # Now compute BFS from a central-ish node to understand distance distribution
    print("\nComputing distance distribution from a peripheral node...")
    src = best_pair[0]
    _, _, dist = bfs_eccentricity(adj, src, n)
    
    # Distance histogram
    max_d = max(d for d in dist if d >= 0)
    hist = defaultdict(int)
    for d in dist:
        if d >= 0:
            hist[d] += 1
    
    print(f"  Source node: {src}")
    print(f"  Max distance: {max_d}")
    print(f"\n  Distance distribution (hop count -> number of facets):")
    for d in sorted(hist.keys()):
        bar = '#' * min(hist[d] // 20, 80)
        print(f"    {d:4d}: {hist[d]:6d}  {bar}")
    
    # Also compute from a more central node
    print("\nComputing eccentricities from 20 random nodes to refine diameter...")
    eccentricities = []
    random_nodes = random.sample(range(n), min(20, n))
    for src in random_nodes:
        ecc, _, _ = bfs_eccentricity(adj, src, n)
        eccentricities.append(ecc)
    
    print(f"  Eccentricities: min={min(eccentricities)}, max={max(eccentricities)}, avg={sum(eccentricities)/len(eccentricities):.1f}")
    print(f"  Refined diameter estimate (max eccentricity): {max(eccentricities)}")
    
    # Summary for dilation array design
    diam = max(diameter_estimate, max(eccentricities))
    print(f"\n{'='*60}")
    print(f"SUMMARY FOR DILATION ARRAY DESIGN")
    print(f"{'='*60}")
    print(f"  Number of facets (nodes): {n}")
    print(f"  Graph diameter: {diam}")
    print(f"  Avg eccentricity: {sum(eccentricities)/len(eccentricities):.1f}")
    print(f"")
    print(f"  For a GAT with L layers and dilation array D=[d1,d2,...,dm],")
    print(f"  the effective receptive field radius after L layers is:")
    print(f"    R = L * max(D)")
    print(f"  For full-face coverage, we need R >= diameter = {diam}")
    print(f"")
    print(f"  Example configurations for full coverage:")
    
    for L in [2, 3, 4]:
        min_max_d = -(-diam // L)  # ceiling division
        print(f"")
        print(f"  L={L} layers:")
        print(f"    Need max(D) >= ceil({diam}/{L}) = {min_max_d}")
        
        # Suggest exponentially growing dilation arrays
        # The idea: cover distances 1, ~sqrt(diam), ~diam/L
        import math
        
        if L == 2:
            suggestions = [
                [1, min_max_d],
                [1, 3, min_max_d],
                [1, int(math.sqrt(min_max_d)), min_max_d],
            ]
        elif L == 3:
            suggestions = [
                [1, min_max_d],
                [1, int(round(min_max_d**0.5)), min_max_d],
                [1, int(round(min_max_d**(1/3))), int(round(min_max_d**(2/3))), min_max_d],
            ]
        elif L == 4:
            suggestions = [
                [1, min_max_d],
                [1, int(round(min_max_d**0.5)), min_max_d],
                [1, int(round(min_max_d**(1/3))), int(round(min_max_d**(2/3))), min_max_d],
            ]
        
        for s in suggestions:
            eff = L * max(s)
            print(f"    D={s}  -> effective radius = {L}*{max(s)} = {eff} {'[OK]' if eff >= diam else '[FAIL]'}")
    
    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()
