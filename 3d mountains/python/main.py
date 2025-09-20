import random
import numpy as np
from numba import jit
from opensimplex import OpenSimplex

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# ──────────────────────────── window / GL setup ────────────────────────────────
pygame.init()
WIDTH, HEIGHT = 1024, 768
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
clock = pygame.time.Clock()

glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)

# ─────────────────────────── terrain / noise params ───────────────────────────
TERRAIN_SIZE = 512          # grid resolution
SCALE        = 0.002        # world-space size per step
OCTAVES      = 8
PERSISTENCE  = 0.5
LACUNARITY   = 2.0

# ─────────────────────────────── camera params ────────────────────────────────
camera_distance = 1000
camera_angle_x  = 30
camera_angle_y  = 0

# ───────────────────────── realistic color palette ────────────────────────────
PALETTE = {
    "deep_water":  np.array([  6,  26,  66]) / 255,
    "shallow":     np.array([ 30,  70, 110]) / 255,
    "beach":       np.array([194, 178, 128]) / 255,
    "grass_low":   np.array([ 80, 120,  60]) / 255,
    "grass_high":  np.array([120, 150,  90]) / 255,
    "forest":      np.array([ 34,  80,  34]) / 255,
    "rock":        np.array([120, 120, 120]) / 255,
    "snow":        np.array([236, 240, 244]) / 255,
}

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def get_biome_color(height, moisture, snow_noise, slope):
    """
    height   ∈ [0,1] before 200× height scale
    moisture ∈ [0,1]
    snow_noise ∈ [0,1]  extra variation near snow line
    slope    ∈ [0,1]  0 = flat, 1 = vertical wall
    """
    # water
    if height < 0.05:
        t = smoothstep(0.00, 0.05, height)
        return (1 - t) * PALETTE["deep_water"] + t * PALETTE["shallow"]

    # beach
    if height < 0.08:
        return PALETTE["beach"]

    # rocky factor from steepness
    rock_factor = smoothstep(0.4, 0.8, slope)

    # lowlands
    if height < 0.45:
        grass = (1 - moisture) * PALETTE["grass_low"] + moisture * PALETTE["forest"]
        return (1 - rock_factor) * grass + rock_factor * PALETTE["rock"]

    # uplands / alpine meadow
    if height < 0.70:
        t = smoothstep(0.45, 0.70, height)
        grass = (1 - t) * PALETTE["grass_high"] + t * PALETTE["rock"]
        return (1 - rock_factor) * grass + rock_factor * PALETTE["rock"]

    # snow
    snow_line = 0.70 + snow_noise * 0.07
    if height > snow_line:
        return PALETTE["snow"]

    # rock-to-snow transition
    t = smoothstep(snow_line - 0.10, snow_line, height)
    return (1 - t) * PALETTE["rock"] + t * PALETTE["snow"]

# ───────────────────────────── helper kernels ─────────────────────────────────
@jit(nopython=True, parallel=True)
def apply_erosion(terrain, iterations=50_000):
    eroded = terrain.copy()
    for _ in range(iterations):
        x  = random.randint(1, terrain.shape[0] - 2)
        y  = random.randint(1, terrain.shape[1] - 2)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            diff = eroded[y, x] - eroded[ny, nx]
            if diff > 0:
                amt = min(diff * 0.1, 0.5)
                eroded[y, x] -= amt
                eroded[ny, nx] += amt
    return eroded

@jit(nopython=True)
def calculate_normal(v1, v2, v3, v4):
    u = v2 - v1
    v = v4 - v1
    n = np.cross(u, v)
    return n / np.linalg.norm(n)

@jit(nopython=True)
def apply_simple_lighting(color, normal):
    light_dir = np.array([0.4, 1.0, 0.3])
    light_dir /= np.linalg.norm(light_dir)

    amb  = 0.35
    diff = max(0.0, np.dot(normal, light_dir))
    hl   = diff * 0.5 + 0.5          # half-Lambert
    return color * np.clip(amb + hl, 0, 1)

# ───────────────────────────── terrain generator ──────────────────────────────
def generate_terrain():
    seed  = random.randint(0, 1_000_000)
    noise = OpenSimplex(seed=seed)

    def noise_at(nx, ny, octaves, persistence, lacunarity):
        val, amp, freq = 0.0, 1.0, 1.0
        for _ in range(octaves):
            val += noise.noise2(nx * freq, ny * freq) * amp
            amp  *= persistence
            freq *= lacunarity
        return val

    h = np.zeros((TERRAIN_SIZE, TERRAIN_SIZE), dtype=np.float32)
    m = np.zeros_like(h)

    for y in range(TERRAIN_SIZE):
        for x in range(TERRAIN_SIZE):
            nx, ny = x * SCALE, y * SCALE

            base  = noise_at(nx, ny, OCTAVES, PERSISTENCE, LACUNARITY)
            mrange = noise_at(nx * 0.1, ny * 0.1, 4, 0.5, 2.0)
            mrange = max(mrange, 0)
            rough  = noise_at(nx * 8, ny * 8, 2, 0.5, 2.0) * 0.05
            comb   = base * 0.3 + mrange * 0.7

            h[y, x] = comb + rough
            m[y, x] = noise_at(nx * 0.05, ny * 0.05, 4, 0.5, 2.0)

    # normalize 0-1 then exaggerate
    h = (h - h.min()) / (h.max() - h.min())
    m = (m - m.min()) / (m.max() - m.min())
    h *= h ** 0 * 0 + 1.0           # no gamma, simpler
    h *= 200.0                      # vertical scale

    # basic hydraulic erosion
    for _ in range(3):
        h = apply_erosion(h)

    # snow-noise field (0-1)
    snow = np.zeros_like(h)
    for y in range(TERRAIN_SIZE):
        for x in range(TERRAIN_SIZE):
            snow[y, x] = noise.noise2(x * SCALE * 4, y * SCALE * 4)
    snow = (snow - snow.min()) / (snow.max() - snow.min())

    return h, m, snow

# ───────────────────────────── geometry baking ────────────────────────────────
print("Generating terrain …")
terrain, moisture, snow_noise = generate_terrain()
print("Terrain ready.")

vertices, colors = [], []

def add_quad(v1, v2, v3, v4, normal):
    slope = 1.0 - abs(normal[1])          # 0 = flat
    # heights to [0,1] for shader
    def nv(v): return v / 200.0

    y1, y2, y3, y4 = nv(v1[1]), nv(v2[1]), nv(v3[1]), nv(v4[1])

    # we look up moisture / snow via grid indices stored in original loop
    # (caller will pass ready-made c1..c4)
    vertices.extend([v1, v2, v3, v4])
    return slope

# grid quads
for i in range(TERRAIN_SIZE - 1):
    for j in range(TERRAIN_SIZE - 1):
        x, z = i - TERRAIN_SIZE // 2, j - TERRAIN_SIZE // 2
        y1, y2, y3, y4 = terrain[i,j], terrain[i+1,j], terrain[i+1,j+1], terrain[i,j+1]

        v1 = np.array([x,     y1, -(z)])
        v2 = np.array([x + 1, y2, -(z)])
        v3 = np.array([x + 1, y3, -(z + 1)])
        v4 = np.array([x,     y4, -(z + 1)])

        n = calculate_normal(v1, v2, v3, v4)
        slope = 1.0 - abs(n[1])

        c1 = get_biome_color(y1/200, moisture[i,j],     snow_noise[i,j],     slope)
        c2 = get_biome_color(y2/200, moisture[i+1,j],   snow_noise[i+1,j],   slope)
        c3 = get_biome_color(y3/200, moisture[i+1,j+1], snow_noise[i+1,j+1], slope)
        c4 = get_biome_color(y4/200, moisture[i,j+1],   snow_noise[i,j+1],   slope)

        c1, c2, c3, c4 = (apply_simple_lighting(c1, n),
                          apply_simple_lighting(c2, n),
                          apply_simple_lighting(c3, n),
                          apply_simple_lighting(c4, n))

        vertices.extend([v1, v2, v3, v4])
        colors.extend([c1, c2, c3, c4])

# side walls (double-sided faces) — uses rock color shaded
ROCK = PALETTE["rock"]
def add_wall(v1, v2, v3, v4):
    n = calculate_normal(v1, v2, v3, v4)
    rc = apply_simple_lighting(ROCK, n)
    vertices.extend([v1, v2, v3, v4,  v4, v3, v2, v1])
    colors.extend([rc] * 8)

for i in range(TERRAIN_SIZE - 1):
    # front (+z)
    x, z = i - TERRAIN_SIZE // 2, TERRAIN_SIZE // 2
    y1, y2 = terrain[i,-1], terrain[i+1,-1]
    add_wall(np.array([x,     y1, -z]),
             np.array([x + 1, y2, -z]),
             np.array([x + 1, 0,  -z]),
             np.array([x,     0,  -z]))
    # back (−z)
    z = -(TERRAIN_SIZE // 2)
    y1, y2 = terrain[i,0], terrain[i+1,0]
    add_wall(np.array([x,     y1, -z]),
             np.array([x + 1, y2, -z]),
             np.array([x + 1, 0,  -z]),
             np.array([x,     0,  -z]))

for j in range(TERRAIN_SIZE - 1):
    # left (−x)
    x, z = -(TERRAIN_SIZE // 2), j - TERRAIN_SIZE // 2
    y1, y2 = terrain[0,j], terrain[0,j+1]
    add_wall(np.array([x,  y1,  -z]),
             np.array([x,  y2,  -(z + 1)]),
             np.array([x,  0,   -(z + 1)]),
             np.array([x,  0,   -z]))
    # right (+x)
    x = TERRAIN_SIZE // 2
    y1, y2 = terrain[-1,j], terrain[-1,j+1]
    add_wall(np.array([x,  y1,  -z]),
             np.array([x,  y2,  -(z + 1)]),
             np.array([x,  0,   -(z + 1)]),
             np.array([x,  0,   -z]))

vertices = np.array(vertices, dtype=np.float32)
colors   = np.array(colors,   dtype=np.float32)
print("Vertex/color buffers built.")

# ───────────────────────────────── main loop ──────────────────────────────────
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == MOUSEWHEEL:
            camera_distance = np.clip(camera_distance - event.y * 20, 100, 2000)

    keys = pygame.key.get_pressed()
    if keys[K_a]: camera_angle_y -= 2
    if keys[K_d]: camera_angle_y += 2

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    gluPerspective(45, WIDTH / HEIGHT, 0.1, 4000.0)
    glTranslatef(0, 0, -camera_distance)
    glRotatef(camera_angle_x, 1, 0, 0)
    glRotatef(camera_angle_y, 0, 1, 0)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glColorPointer(3, GL_FLOAT, 0, colors)
    glDrawArrays(GL_QUADS, 0, len(vertices))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

    pygame.display.flip()
    clock.tick(60)
    print(f"\rFPS: {clock.get_fps():.1f}", end='', flush=True)

pygame.quit()
