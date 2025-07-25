import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from opensimplex import OpenSimplex
import random
from numba import jit

# Initialize Pygame and OpenGL
pygame.init()
width, height = 1024, 768
pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
clock = pygame.time.Clock()
ROCK = np.array([0.5, 0.5, 0.5])  # Define ROCK color
# Terrain settings
terrain_size = 1014
scale = 0.002
octaves = 8
persistence = 0.5
lacunarity = 2.0

# Camera settings
camera_distance = 1000
camera_angle_x = 30
camera_angle_y = 0

# Sun settings
sun_direction = np.array([0.5, 1, 0.5])  # Initial sun direction

@jit(nopython=True, parallel=True)
def apply_erosion(terrain, iterations=60000):
    eroded = terrain.copy()
    for _ in range(iterations):
        x, y = random.randint(1, terrain.shape[0]-2), random.randint(1, terrain.shape[1]-2)
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            diff = eroded[y, x] - eroded[ny, nx]
            if diff > 0:
                amount = min(diff * 0.1, 0.5)
                eroded[y, x] -= amount
                eroded[ny, nx] += amount
    return eroded

@jit(nopython=True)
def calculate_normal(v1, v2, v3, v4):
    u = v2 - v1
    v = v4 - v1
    normal = np.cross(u, v)
    return normal / np.linalg.norm(normal)

def get_biome_color(height, moisture, snow_noise):
    SNOW = np.array([0.9, 0.9, 0.9])
    ROCK = np.array([0.5, 0.5, 0.5])
    GRASS = np.array([0.3, 0.4, 0.3])
    FOREST = np.array([0.1, 0.3, 0.1])
    WATER = np.array([0.0, 0.1, 0.2])

    height = max(0, min(1, height))
    moisture = max(0, min(1, moisture))
    snow_threshold = 0.7 + snow_noise * 0.1 + np.random.uniform(-0.05, 0.05)

    if height > snow_threshold:
        return SNOW
    elif height > 0.5:
        t = (height - 0.5) / (snow_threshold - 0.5)
        return ROCK * t + GRASS * (1 - t)
    elif height > 0.1:
        if moisture > 0.6:
            return FOREST
        else:
            t = (moisture - 0.3) / 0.3
            return FOREST * t + GRASS * (1 - t)
    else:
        return WATER

@jit(nopython=True)
def apply_simple_lighting(color, normal, sun_dir):
    sun_dir = sun_dir / np.linalg.norm(sun_dir)
    ambient = 0.3
    diffuse = max(0, np.dot(normal, sun_dir))
    shadow_strength = 0.5 if diffuse < 0.1 else 1  # Simple shadow effect
    return color * (ambient + diffuse * 0.7) * shadow_strength

def generate_terrain():
    seed = random.randint(0, 1000000)
    noise = OpenSimplex(seed=seed)

    def noise_at(nx, ny, octaves, persistence, lacunarity):
        value = 0
        amplitude = 1
        frequency = 1
        for _ in range(octaves):
            value += noise.noise2(nx * frequency, ny * frequency) * amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return value

    terrain = np.zeros((terrain_size, terrain_size), dtype=np.float32)
    moisture = np.zeros((terrain_size, terrain_size), dtype=np.float32)
    for y in range(terrain_size):
        for x in range(terrain_size):
            nx = x * scale
            ny = y * scale
            
            base = noise_at(nx, ny, octaves, persistence, lacunarity)
            base = (base + 1) / 2
            base = np.power(base, 1.2)
            
            mountain_range = noise_at(nx * 0.02, ny * 0.02, 2, 0.5, 2.0)
            mountain_range = np.maximum(mountain_range, 0)
            mountain_range = np.power(mountain_range, 3)
            
            combined = base * 0.3 + mountain_range * 0.7
            roughness = noise_at(nx * 8, ny * 8, 2, 0.5, 2.0) * 0.00001
            
            terrain[y, x] = combined + roughness
            moisture[y, x] = noise_at(nx * 0.05, ny * 0.05, 4, 0.5, 2.0)

    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    moisture = (moisture - moisture.min()) / (moisture.max() - moisture.min())
    terrain = np.power(terrain, 1.3) * 250
    terrain = apply_erosion(terrain, iterations=20000)

    snow_noise = np.zeros((terrain_size, terrain_size), dtype=np.float32)
    for y in range(terrain_size):
        for x in range(terrain_size):
            snow_noise[y, x] = noise.noise2(x * scale * 4, y * scale * 4)
    
    snow_noise = (snow_noise - snow_noise.min()) / (snow_noise.max() - snow_noise.min())
    
    return terrain, moisture, snow_noise

print("Generating terrain...")
terrain, moisture, snow_noise = generate_terrain()
print("Terrain generated.")

# Create vertex and color arrays
print("Creating vertex and color arrays...")
vertices = []
colors = []
for i in range(terrain_size - 1):
    for j in range(terrain_size - 1):
        x, z = i - terrain_size // 2, j - terrain_size // 2
        y1, y2, y3, y4 = terrain[i, j], terrain[i + 1, j], terrain[i + 1, j + 1], terrain[i, j + 1]
        v1 = np.array([x, y1, -z])
        v2 = np.array([x + 1, y2, -z])
        v3 = np.array([x + 1, y3, -(z + 1)])
        v4 = np.array([x, y4, -(z + 1)])
        
        normal = calculate_normal(v1, v2, v3, v4)
        
        c1 = apply_simple_lighting(get_biome_color(y1 / 200, moisture[i, j], snow_noise[i, j]), normal, sun_direction)
        c2 = apply_simple_lighting(get_biome_color(y2 / 200, moisture[i + 1, j], snow_noise[i + 1, j]), normal, sun_direction)
        c3 = apply_simple_lighting(get_biome_color(y3 / 200, moisture[i + 1, j + 1], snow_noise[i + 1, j + 1]), normal, sun_direction)
        c4 = apply_simple_lighting(get_biome_color(y4 / 200, moisture[i, j + 1], snow_noise[i, j + 1]), normal, sun_direction)
        
        vertices.extend([v1, v2, v3, v4])
        colors.extend([c1, c2, c3, c4])

vertices = np.array(vertices, dtype=np.float32)
colors = np.array(colors, dtype=np.float32)
print("Arrays created.")

# Set up OpenGL
glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)
glCullFace(GL_BACK)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            camera_distance -= event.y * 20
            camera_distance = max(100, min(2000, camera_distance))

    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        camera_angle_y -= 2
    if keys[pygame.K_d]:
        camera_angle_y += 2
    if keys[pygame.K_UP]:
        sun_direction[1] += 0.1  # Increase sun height
    if keys[pygame.K_DOWN]:
        sun_direction[1] -= 0.1  # Decrease sun height
    if keys[pygame.K_LEFT]:
        sun_direction[0] -= 0.1  # Move sun left
    if keys[pygame.K_RIGHT]:
        sun_direction[0] += 0.1  # Move sun right

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Set up camera
    gluPerspective(45, (width / height), 0.1, 4000.0)
    glTranslatef(0, 0, -camera_distance)
    glRotatef(camera_angle_x, 1, 0, 0)
    glRotatef(camera_angle_y, 0, 1, 0)

    # Render terrain
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glColorPointer(3, GL_FLOAT, 0, colors)
    glDrawArrays(GL_QUADS, 0, len(vertices))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)

    pygame.display.flip()
    clock.tick(60)
    print(f"\rFPS: {clock.get_fps():.2f}", end='', flush=True)

pygame.quit()
