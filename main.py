import neat
import pygame
import sys
import random
import os
import pickle

# Trying to create a flappy bird game and use NEAT to train AI to play the game
# By Leo

# pygame init functions
pygame.init()
pygame.display.set_caption("Flappy Bird With NEAT By Leo")
window = pygame.display.set_mode((288, 512))
clock = pygame.time.Clock()

# load images
bird_image = pygame.image.load('assets/bluebird-midflap.png').convert()
base_image = pygame.image.load('assets/base.png').convert()
pipe_image = pygame.image.load('assets/pipe-green.png').convert()
background_image = pygame.image.load('assets/background-night.png').convert()

# create constant surfaces
background_surface = background_image

# game variables
current_score = 0
highest_score = 0
gen = 0

# game constants
game_font = pygame.font.Font('04B_19__.TTF', 20)
gravity = 0.5


class Bird:
    """
    Creates the birds
    """
    bird_image = bird_image

    def __init__(self):
        """
        initializes a bird with x and y values
        """
        self.velocity = 0
        self.surface = bird_image
        self.rect = self.surface.get_rect(center=(50, 256))

    def move(self):
        """
        makes the bird move at all times
        :return:
        """
        self.velocity += gravity
        self.rect.y += self.velocity

    def jump(self):
        """
        jump method (active)
        :return: None
        """
        self.velocity = -7
        self.rect.y += self.velocity

    def collision(self, pipe):
        """
        detects collision with the pipe and the base
        returns if collide
        :return:
        """
        if self.rect.colliderect(pipe.top_rect):
            return True
        elif self.rect.colliderect(pipe.bottom_rect):
            return True
        elif self.rect.y >= 450 or self.rect.y <= 0:
            return True

    def animation(self):
        """
        gives the bird animation
        :return:a new bird surface
        """
        new_bird_surface = pygame.transform.rotozoom(self.surface, self.velocity * -5, 1)
        return new_bird_surface

    def draw(self):
        """
        draws the bird
        :return:
        """
        window.blit(self.animation(), self.rect)


class Pipe:
    """
    creates a pair of pipes
    generate pipes after the first one has been passed
    """
    gap = 100
    pipe_image = pipe_image
    velocity_x = 5
    # gap for the two pipes

    def __init__(self):
        """
        initiates the two pipes by granting its location
        """
        self.x = 300
        self.bottom_pipe_y, self.top_pipe_y = self.pipe_location_generator()

        self.bottom_surface = pipe_image
        self.bottom_rect = self.bottom_surface.get_rect(midtop=(self.x, self.bottom_pipe_y))

        self.top_surface = pygame.transform.flip(pipe_image, False, True)
        self.top_rect = self.top_surface.get_rect(midbottom=(self.x, self.top_pipe_y))

        self.passed = False

    def pipe_location_generator(self):
        """
        gives the pipe its location
        choose between three different locations
        :return: new top and bottom pipe values
        """
        pipe_height = random.randrange(200, 430)
        bottom = pipe_height
        top = bottom - self.gap
        return bottom, top

    def draw(self):
        self.move()
        window.blit(self.top_surface, self.top_rect)
        window.blit(self.bottom_surface, self.bottom_rect)

    def move(self):
        self.x -= self.velocity_x
        self.bottom_rect.x = self.x
        self.top_rect.x = self.x

    def is_passed(self, bird):
        if bird.x > self.x:
            return True

    def is_out_of_frame(self):
        if self.x < -40:
            return True


class Base:
    """
    creates the two bases
    """
    image = base_image
    y = 450

    def __init__(self):
        """
        initiate a base
        """
        self.x = 0
        self.velocity = 5
        self.base_surface = self.image

    def draw(self):
        self.move()
        window.blit(self.base_surface, (self.x, self.y))
        window.blit(self.base_surface, (self.x + 288, self.y))

    def move(self):
        self.x -= self.velocity
        if self.x < -288:
            self.x = 0


def draw_background():
    window.blit(background_surface, (0, 0))


def show_score(gen, alive):
    score_surface = game_font.render("Score:{}".format(current_score), False, (255, 255, 255))
    score_surface_rect = score_surface.get_rect(center=(210, 40))
    window.blit(score_surface, score_surface_rect)

    generation_surface = game_font.render("Generation:{}".format(gen), False, (255, 255, 255))
    generation_surface_rect = score_surface.get_rect(center=(50, 40))
    window.blit(generation_surface, generation_surface_rect)

    alive_surface = game_font.render("Birds Alive:{}".format(alive), False, (255, 255, 255))
    alive_surface_rect = score_surface.get_rect(center=(50, 60))
    window.blit(alive_surface, alive_surface_rect)


def eval_genome(genomes, config):
    global gen
    global current_score
    current_score = 0
    # fitness function - measures and evaluates how far a bird goes
    # basically generates a game with the genome given
    nets = []
    ge = []
    birds = []
    gen += 1
    # keeps track of each bird with its own network and genome
    for genome_id, genome in genomes:
        # setting up a nero network for the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        genome.fitness = 0
        # each genome has a fitness
        ge.append(genome)

    pipe = Pipe()
    base = Base()
    run = True

    while run and len(birds) > 0:
        # the action loop
        draw_background()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if len(birds) <= 0:
            # if no birds is alive break out of the while loop
            break

        for x, bird in enumerate(birds):
            # activate nero network for each bird
            bird.move()
            bird.draw()
            ge[x].fitness += 0.1
            output = nets[x].activate((bird.rect.y, abs(bird.rect.y - pipe.top_pipe_y), abs(bird.rect.y - pipe.bottom_pipe_y)))
            # net.active(inputs)
            if output[0] > 0.5:
                # the one output neuron's value
                bird.jump()

        for x, bird in enumerate(birds):
            # combination of x and enumerate allows you to get the position
            if bird.collision(pipe):
                ge[x].fitness -= 1
                # if a bird hits a pipe, it will have less fitness
                birds.pop(x)
                ge.pop(x)
                nets.pop(x)
                # removes the dead birds from population

        if pipe.is_out_of_frame():
            # when the bird make it here increase fitness if bird goes pass a pipe
            current_score += 1
            for g in ge:
                g.fitness += 2
            pipe = Pipe()

        if current_score > 30:
            pickle.dump(nets[0], open("best.pickle", "wb"))

        pipe.draw()
        base.draw()
        show_score(gen, len(birds))
        pygame.display.update()
        clock.tick(60)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_file)
    # set all the properties from the config file
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    # shows stats for each generation (output)
    population.add_reporter(neat.StatisticsReporter())
    winner = population.run(eval_genome, 100)
    # runs the game 50 times


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    # gives the path to the directory we are currently in
    config_path = os.path.join(local_dir, "config.feedforward.txt")
    # finding the path to the file
    run(config_path)