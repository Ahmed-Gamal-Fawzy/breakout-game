import pygame
import random
import itertools
import math

# Constants
SIZE = (424, 430)
HEIGHT_OF_BRICK = 13
WIDTH_OF_BRICK = 32
PADDLE_HEIGHT = 8
PADDLE_WIDTH = 100
BALL_DIAMETER = 12
BALL_RADIUS = BALL_DIAMETER // 2
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 120, 255)
GREEN = (0, 255, 120)
RED = (255, 60, 60)
YELLOW = (255, 215, 0)
PURPLE = (138, 43, 226)
ORANGE = (255, 165, 0)

LEVEL_COLORS = [BLUE, GREEN, RED, YELLOW]

pygame.init()
base_surf = pygame.Surface(SIZE)
zoom_level = 1.0
screen = pygame.display.set_mode((int(SIZE[0] * zoom_level), int(SIZE[1] * zoom_level)))
pygame.display.set_caption("BREAKOUT")
clock = pygame.time.Clock()

# Game state
level_unlocked = {1: True, 2: True, 3: True, 4: True}
current_level = 1
hover_level = 0
particles = []

class Particle:
    def __init__(self):
        self.x = random.randint(0, SIZE[0])
        self.y = random.randint(0, SIZE[1])
        self.speed = random.uniform(0.5, 1.5)
        self.size = random.randint(1, 3)
    
    def update(self):
        self.y += self.speed
        if self.y > SIZE[1]:
            self.y = 0
            self.x = random.randint(0, SIZE[0])

def create_particles():
    global particles
    particles = [Particle() for _ in range(50)]

def draw_menu():
    global hover_level
    base_surf.fill(BLACK)
    
    # Draw star particles
    for p in particles:
        pygame.draw.circle(base_surf, WHITE, (int(p.x), int(p.y)), p.size)
        p.update()
    
    # Animated title
    title_font = pygame.font.Font(None, 48)
    time = pygame.time.get_ticks() / 1000
    y_offset = 10 * math.sin(time * 2)
    title_text = title_font.render("SELECT LEVEL", True, WHITE)
    title_rect = title_text.get_rect(center=(SIZE[0]//2, 60 + y_offset))
    base_surf.blit(title_text, title_rect)
    
    # Level buttons
    level_buttons = []
    button_size = 90
    padding = 20
    start_x = (SIZE[0] - (2 * button_size + padding)) // 2
    start_y = 120
    
    for i in range(4):
        row = i // 2
        col = i % 2
        x = start_x + col * (button_size + padding)
        y = start_y + row * (button_size + padding)
        rect = pygame.Rect(x, y, button_size, button_size)
        level_buttons.append((rect, i+1))
        
        # Hover effect
        hover = rect.collidepoint(pygame.mouse.get_pos())
        if hover:
            hover_level = i+1
            glow_size = button_size + 10
            glow_rect = pygame.Rect(x-5, y-5, glow_size, glow_size)
            alpha = abs(math.sin(time * 4)) * 100
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*LEVEL_COLORS[i], alpha), (0, 0, glow_size, glow_size), border_radius=10)
            base_surf.blit(shape_surf, glow_rect)
        
        # Button background
        color = LEVEL_COLORS[i]
        pygame.draw.rect(base_surf, color, rect, border_radius=10)
        
        # Level number
        font = pygame.font.Font(None, 36)
        text_color = WHITE if hover else (200, 200, 200)
        level_text = font.render(str(i+1), True, text_color)
        text_rect = level_text.get_rect(center=rect.center)
        base_surf.blit(level_text, text_rect)
        
        # Animated border
        if hover:
            border_width = abs(int(math.sin(time * 5) * 3))
            pygame.draw.rect(base_surf, WHITE, rect, border_width, border_radius=10)
    
    scaled_surf = pygame.transform.scale(base_surf, screen.get_size())
    screen.blit(scaled_surf, (0, 0))
    pygame.display.update()
    return level_buttons

def select_level():
    global current_level, hover_level, zoom_level
    create_particles()
    while True:
        level_buttons = draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, level in level_buttons:
                    if rect.collidepoint(event.pos):
                        current_level = level
                        return
            if event.type == pygame.MOUSEMOTION:
                hover_level = 0
                for rect, level in level_buttons:
                    if rect.collidepoint(event.pos):
                        hover_level = level
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS):
                    zoom_level = min(2.0, zoom_level + 0.1)
                    screen = pygame.display.set_mode((int(SIZE[0] * zoom_level), int(SIZE[1] * zoom_level)))
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom_level = max(0.5, zoom_level - 0.1)
                    screen = pygame.display.set_mode((int(SIZE[0] * zoom_level), int(SIZE[1] * zoom_level)))
        
        clock.tick(FPS)

class Breakout:
    def __init__(self, level):
        self.level = level
        self.ball_speed = 3 + (level - 1)
        self.paddle = pygame.Rect(215, SIZE[1] - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(225, self.paddle.top - BALL_DIAMETER, BALL_DIAMETER, BALL_DIAMETER)
        self.ball_vel = [self.ball_speed, -self.ball_speed]
        self.create_bricks()
    
    def create_bricks(self):
        y_ofs = 20
        self.bricks = []
        rows = 5 + (self.level - 1) * 2
        for i in range(rows):
            x_ofs = 15
            for j in range(10):
                self.bricks.append(pygame.Rect(x_ofs, y_ofs, WIDTH_OF_BRICK, HEIGHT_OF_BRICK))
                x_ofs += WIDTH_OF_BRICK + 1
            y_ofs += HEIGHT_OF_BRICK + 1
    
    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.paddle.left -= 6
            if self.paddle.left < 0:
                self.paddle.left = 0
        if keys[pygame.K_RIGHT]:
            self.paddle.left += 6
            if self.paddle.left > SIZE[0] - PADDLE_WIDTH:
                self.paddle.left = SIZE[0] - PADDLE_WIDTH
        
        self.ball.left += self.ball_vel[0]
        self.ball.top += self.ball_vel[1]
        
        if self.ball.left <= 0 or self.ball.left >= SIZE[0] - BALL_DIAMETER:
            self.ball_vel[0] = -self.ball_vel[0]
        if self.ball.top <= 0:
            self.ball_vel[1] = -self.ball_vel[1]
        if self.ball.top >= SIZE[1] - BALL_DIAMETER:
            return False
        
        for brick in self.bricks:
            if self.ball.colliderect(brick):
                self.bricks.remove(brick)
                self.ball_vel[1] = -self.ball_vel[1]
                break
        
        if self.ball.colliderect(self.paddle):
            self.ball.top = self.paddle.top - BALL_DIAMETER
            self.ball_vel[1] = -self.ball_vel[1]
        
        return True
    
    def draw(self):
        base_surf.fill(BLACK)
        for brick in self.bricks:
            pygame.draw.rect(base_surf, (153, 76, 0), brick)
        pygame.draw.circle(base_surf, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)
        pygame.draw.rect(base_surf, (204, 0, 0), self.paddle)
        scaled_surf = pygame.transform.scale(base_surf, screen.get_size())
        screen.blit(scaled_surf, (0, 0))
        pygame.display.update()
    
    def is_won(self):
        return len(self.bricks) == 0

def run_game():
    global zoom_level
    while True:
        select_level()
        game = Breakout(current_level)
        running = True
        while running:
            keys = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS):
                        zoom_level = min(2.0, zoom_level + 0.1)
                        screen = pygame.display.set_mode((int(SIZE[0] * zoom_level), int(SIZE[1] * zoom_level)))
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        zoom_level = max(0.5, zoom_level - 0.1)
                        screen = pygame.display.set_mode((int(SIZE[0] * zoom_level), int(SIZE[1] * zoom_level)))
            
            if not game.update(keys):
                running = False
            game.draw()
            clock.tick(FPS)
        
if __name__ == "__main__":
    run_game()