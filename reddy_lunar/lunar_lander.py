import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding

import pyglet

from copy import copy

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# Too see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v0
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

GROUND_SLACK = 0.01
LANDED_SLACK = 50

MAX_NUM_STEPS = 1000

N_OBS_DIM = 9
N_ACT_DIM = 6 # num discrete actions

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40 # 40 is too difficult for human players, 400 a bit easier

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.lander==contact.fixtureA.body or self.env.lander==contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class LunarLander(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        #'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

        high = np.array([np.inf]*N_OBS_DIM)  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.fps = FPS
        self.curr_step = None
        self.curr_obs = None
        self.goal = None
        self.fan = None
        self.fan_confs = None
        self.at_site = None
        self.grounded = None

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def _reset(self):
        self.trajectory = []
        self.actions = []
        self.curr_step = 0
        self.curr_obs = None
        self.fan = False
        self.fan_confs = []
        self.at_site = [False for _ in range(LANDED_SLACK)]
        self.grounded = [False for _ in range(LANDED_SLACK)]

        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,) )
        chunk_x  = [W/(CHUNKS-1)*i for i in range(CHUNKS)]

        #helipad_chunk = CHUNKS // 2
        # randomize helipad x coord
        helipad_chunk = self.goal if self.goal is not None else np.random.choice(range(1, CHUNKS-1))

        self.helipad_x1 = chunk_x[helipad_chunk-1]
        self.helipad_x2 = chunk_x[helipad_chunk+1]
        self.helipad_y  = H/4
        height[helipad_chunk-2] = self.helipad_y
        height[helipad_chunk-1] = self.helipad_y
        height[helipad_chunk+0] = self.helipad_y
        height[helipad_chunk+1] = self.helipad_y
        height[helipad_chunk+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody( shapes=edgeShape(vertices=[(0, 0), (W, 0)]) )
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i],   smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append( [p1, p2, (p2[0],H), (p1[0],H)] )

        self.moon.color1 = (0.0,0.0,0.0)
        self.moon.color2 = (0.0,0.0,0.0)

        initial_y = VIEWPORT_H/SCALE#*0.75
        self.lander = self.world.CreateDynamicBody(
            position = (VIEWPORT_W/SCALE/2, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.lander.color1 = (0.5,0.4,0.9)
        self.lander.color2 = (0.3,0.3,0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5,0.4,0.9)
            leg.color2 = (0.3,0.3,0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i==-1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self._step(np.array([0,0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x,y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0,0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl<0):
            self.world.DestroyBody(self.particles.pop(0))

    def sim_step(
        self, action,
        fps=FPS, mep=MAIN_ENGINE_POWER, sep=SIDE_ENGINE_POWER,
        sea=SIDE_ENGINE_AWAY, seh=SIDE_ENGINE_HEIGHT, scale=SCALE, leg_down=LEG_DOWN):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

        bsu_pos_x = copy(self.lander.position.x)
        bsu_pos_y = copy(self.lander.position.y)
        bsu_ang = copy(self.lander.angle)
        bsu_vel_x = copy(self.lander.linearVelocity.x)
        bsu_vel_y = copy(self.lander.linearVelocity.y)
        bsu_angvel = copy(self.lander.angularVelocity)
        bsu_awake = copy(self.lander.awake)
        bsu_game_over = copy(self.game_over)
        bsu_ground_contacts = copy([self.legs[i].ground_contact for i in range(2)])

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / scale for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox =  tip[0]*(4/scale + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/scale + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            self.lander.ApplyLinearImpulse( (-ox*mep*m_power, -oy*mep*m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1,3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*sea/scale)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*sea/scale)
            impulse_pos = (self.lander.position[0] + ox - tip[0]*17/scale, self.lander.position[1] + oy + tip[1]*seh/scale)
            self.lander.ApplyLinearImpulse( (-ox*sep*s_power, -oy*sep*s_power), impulse_pos, True)

        # perform slow update
        self.world.Step(1.0/fps, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        helipad_x = (self.helipad_x1 + self.helipad_x2) / 2
        slow_state = [
            (pos.x - VIEWPORT_W/scale/2) / (VIEWPORT_W/scale/2),
            (pos.y - (self.helipad_y+leg_down/scale)) / (VIEWPORT_W/scale/2),
            vel.x*(VIEWPORT_W/scale/2)/self.fps,
            vel.y*(VIEWPORT_H/scale/2)/self.fps,
            self.lander.angle,
            20.0*self.lander.angularVelocity/self.fps,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            (helipad_x - VIEWPORT_W/scale/2) / (VIEWPORT_W/scale/2)
            ]

        reward = 0
        shaping = 0
        dx = (pos.x - helipad_x) / (VIEWPORT_W/SCALE/2)
        shaping += - 100*np.sqrt(slow_state[2]*slow_state[2] + slow_state[3]*slow_state[3]) - 100*abs(slow_state[4])
        shaping += -100*np.sqrt(dx*dx + slow_state[1]*slow_state[1]) + 10*slow_state[6] + 10*slow_state[7]
        if self.prev_shaping is not None:
          reward = shaping - self.prev_shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power*0.03

        oob = abs(slow_state[0]) >= 1.0
        timeout = self.curr_step + 1 >= MAX_NUM_STEPS
        not_awake = not self.lander.awake

        at_site = copy(self.at_site)
        at_site[:-1] = at_site[1:]
        at_site[-1] = pos.x >= self.helipad_x1 and pos.x <= self.helipad_x2 and slow_state[1] <= GROUND_SLACK
        grounded = copy(self.grounded)
        grounded[:-1] = grounded[1:]
        grounded[-1] = self.legs[0].ground_contact and self.legs[1].ground_contact
        landed = all(at_site) and any(grounded)

        done = self.game_over or oob or not_awake or timeout or landed
        if done:
          if self.game_over or oob:
            reward = -100
          elif at_site[-1]:
            reward = +100

        # undo slow update
        self.lander.position = (bsu_pos_x, bsu_pos_y)
        self.lander.angle = bsu_ang
        self.lander.linearVelocity = (bsu_vel_x, bsu_vel_y)
        self.lander.angularVelocity = bsu_angvel
        self.lander.awake = bsu_awake
        self.game_over = bsu_game_over
        for i in range(2):
          self.legs[i].ground_contact = bsu_ground_contacts[i]

        return np.array(slow_state), reward

    def _step(self, action):
        info = {}
        if self.fan:
          self.fan = False
          obses, rews = zip(*[self.sim_step(action, **conf) for conf in self.fan_confs])
          info['rews'] = rews
          info['obses'] = obses

        assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0]);
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power>=0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(           ( ox*MAIN_ENGINE_POWER*m_power,  oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1,3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power>=0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0]*17/SCALE, self.lander.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(           ( ox*SIDE_ENGINE_POWER*s_power,  oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

        # perform normal update
        self.world.Step(1.0/self.fps, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        helipad_x = (self.helipad_x1 + self.helipad_x2) / 2
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_W/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/self.fps,
            vel.y*(VIEWPORT_H/SCALE/2)/self.fps,
            self.lander.angle,
            20.0*self.lander.angularVelocity/self.fps,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            (helipad_x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2)
            ]
        assert len(state)==N_OBS_DIM

        self.curr_step += 1
        self.curr_obs = np.array(state)

        reward = 0
        shaping = 0
        dx = (pos.x - helipad_x) / (VIEWPORT_W/SCALE/2)
        shaping += - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) - 100*abs(state[4])
        shaping += -100*np.sqrt(dx*dx + state[1]*state[1]) + 10*state[6] + 10*state[7]
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power*0.03

        oob = abs(state[0]) >= 1.0
        timeout = self.curr_step >= MAX_NUM_STEPS
        not_awake = not self.lander.awake

        self.at_site[:-1] = self.at_site[1:]
        self.at_site[-1] = pos.x >= self.helipad_x1 and pos.x <= self.helipad_x2 and state[1] <= GROUND_SLACK
        self.grounded[:-1] = self.grounded[1:]
        self.grounded[-1] = self.legs[0].ground_contact and self.legs[1].ground_contact
        landed = all(self.at_site) and any(self.grounded)

        done = self.game_over or oob or not_awake or timeout or landed
        if done:
          if self.game_over or oob:
            reward = -100
            self.lander.color1 = (255,0,0)
          elif self.at_site[-1]:
            reward = +100
            self.lander.color1 = (0,255,0)
          elif timeout:
            self.lander.color1 = (255,0,0)

        return np.array(state), reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
            obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0,0,0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(1,1,1) )
            self.viewer.draw_polygon( [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)], color=(0.8,0.8,0) )

        clock_prog = self.curr_step / MAX_NUM_STEPS
        self.viewer.draw_polyline( [(0, 0.05*VIEWPORT_H/SCALE), (clock_prog*VIEWPORT_W/SCALE, 0.05*VIEWPORT_H/SCALE)], color=(255,0,0), linewidth=5  )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class LunarLanderContinuous(LunarLander):
    continuous = True

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
        elif angle_todo < -0.05: a = 3
        elif angle_todo > +0.05: a = 1
    return a

if __name__=="__main__":
    #env = LunarLander()
    env = LunarLanderContinuous()
    s = env.reset()
    total_reward = 0
    steps = 0
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        env.render()
        total_reward += r
        if steps % 20 == 0 or done:
            print(["{:+0.2f}".format(x) for x in s])
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
