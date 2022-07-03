import os
import machine_common_sense as mcs
from os.path import isfile, join
import numpy as np
from PIL import Image
import operator
import cv2
import imutils
from scipy.spatial import distance
from collections import OrderedDict
from .astar_planner import Astar_Planner

left = np.array([-1, 0])
right = np.array([1, 0])
up = np.array([0, 1])
down = np.array([0, -1])
action_list = [left, right, up, down]

class A_star_model:
    def __init__(self, goal_xy, grid):
        self.goal_xy = goal_xy
        self.grid = grid

    def __call__(self, state, action):
        next_x = state[0] + action[0]
        next_y = state[1] + action[1]
        # is out of bounds
        if (next_x > 600 or next_x < 0 or next_y > 400 or next_y < 0):
            return [], action
        # is at an occluder
        if (self.grid[next_x][next_y] == 255):
            return [], action
        else:
            next_state = [next_x, next_y]
        return [next_state], action

def heuristic(state, goal):
    return distance.euclidean(state, goal)

def goal_check(agent_xy, goal_xy):
    if agent_xy == goal_xy:
        return True

def state_to_tuple(state):
    return (state[0], state[1])

class Agency():
    def __init__(self, output):
        self.trial = output.habituation_trial
        self.trial_images = [[] for i in range(8)]
        self.trial_d_maps = [[] for i in range(8)]
        self.test_images = []
        self.test_d_maps = []
        self.plan_lens = []
        self.plan_diff = []
        self.goal_xy = None
        self.agent_xy = None
        self.begin_dist = None
        self.end_dist = None

    def create_grid(self, image, depth_map):
        # creates a 2d (600, 400) grid for A* star
        # removes found objs from grid
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask = self.remove_background(img)
        # uses depth map to identify remaining occluders (walls)
        result = np.zeros([600, 400], dtype=np.uint8)
        img_result = np.zeros([400, 600], dtype=np.uint8)

        for y in range(400):
            norm = depth_map[y][0]
            for x in range(600):
                norm_value = depth_map[y][x] / norm
                # only keep occluders minus objects
                if norm_value != 1:
                    result[x][y] = 255
                    img_result[y][x] = 255
                if mask[y][x] == 255:
                    result[x][y] = 0
                    img_result[y][x] = 0
        return result, img_result

    def remove_background(self, img):
        # filters out grey
        hsv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_grey = np.array([0, 48, 45])
        upper_grey = np.array([255, 255, 255])
        mask_grey = cv2.inRange(hsv_grey, lower_grey, upper_grey)
        result = cv2.bitwise_and(img, img, mask=mask_grey)
        result[mask_grey == 0] = (255, 255, 255)

        # filters out pink starting block
        lower_red = np.array([180, 5, 180])
        upper_red = np.array([255, 150, 255])
        mask_pink = cv2.inRange(result, lower_red, upper_red)
        result[mask_pink > 0] = (255, 255, 255)

        # obtains resulting image with gray binary mask
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

        # sketchy way of removing a weird detection error on left corner of wall
        hack = []
        for row in binary:
            row[0:80] = 0
            hack.append(row)
        hack = np.array(hack)
        hack = hack.astype(np.uint8)
        return hack

    def get_xy(self, contour):
        x = 0
        y = 0
        coord_cur = contour[0]
        for coord in contour:
            if coord[0][1] > y:
                x = coord[0][0]
                y = coord[0][1]
                coord_cur = coord[0]
                if coord[0][0] > x:
                    x = coord[0][0]
                    y = coord[0][1]
                    coord_cur = coord[0]
        return tuple(coord_cur)

    def get_agent_xy(self, x_array, y_array):
        x = x_array[0]
        y = y_array[0]
        for i in range(len(x_array)):
            if y_array[i] >= y:
                x = x_array[i]
                y = y_array[i]
        return (x, y)

    def find_contours(self, mask, img):
        # finds objs (contours) in image, returns in OrderedDict
        shapes = OrderedDict()
        contours = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100:
                cv2.fillPoly(mask, pts=[c], color=0)
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours[0]:
            coord = self.get_xy(contour)
            # add detected shapes to dict
            if coord in shapes.keys():
                shapes[coord].append(contour)
            else:
                shapes[coord] = [contour]
            img = cv2.circle(img, coord, 10, (255, 0, 0), 1)
        return img, shapes

    def dist(self, obj1, obj2):
        return distance.euclidean(obj1[1], obj2[1])

    def find_objs(self, images, d_maps):
        # function to find objs and agent trajs throughout a trial
        imgs = []
        objs = OrderedDict()
        agent_trajs = []

        counter = 0
        for image in images:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            mask = self.remove_background(img)
            img, contours = self.find_contours(mask, img)
            imgs.append(img)

            # adds identified objects to large dict
            for coord in contours.keys():
                if coord in objs.keys():
                    objs[coord].append(contours[coord])
                else:
                    objs[coord] = contours[coord]

            # at beginning, extracts agent from pink block
            if counter == 0:
                # gets agent starting position from pink block
                agent_start = cv2.cvtColor(np.array(images[-1]), cv2.COLOR_RGB2BGR)
                lower_pink = np.array([180, 5, 180])
                upper_pink = np.array([255, 150, 255])
                pink_mask = cv2.inRange(agent_start, lower_pink, upper_pink)
                agent_start, contours = self.find_contours(pink_mask, agent_start)
                for key in contours:
                    agent_trajs.append(key)
            # otherwords, gets from depth difference
            else:
                new_depth = d_maps[counter]
                old_depth = d_maps[counter - 1]
                diff = np.nonzero(new_depth - old_depth)
                if len(diff[0]) > 0:
                    agent_coord = self.get_agent_xy(diff[1], diff[0])
                    agent_trajs.append(agent_coord)
                else:
                    agent_trajs.append(agent_trajs[-1])
            counter += 1

        # strips out objs that aren't in at least 1/2 frames
        for coord in objs.copy().keys():
            if len(objs[coord]) < len(images) / 2:
                objs.pop(coord, None)

        return imgs, objs, agent_trajs

    def find_goal(self, objs, agent_trajs):
        dists_from_agent = OrderedDict()
        goal = None
        for key in objs:
            obj_dist_from_agent = []
            for traj in agent_trajs:
                obj_dist_from_agent.append(self.dist(traj, key))
            dists_from_agent[key] = obj_dist_from_agent

        # chooses goal as obj agent moves towards most frequently
        obj_moving_closer = OrderedDict()
        goal = OrderedDict()
        for key in dists_from_agent:
            count_moving_closer = 0
            dists = dists_from_agent[key]
            for i in range(1, len(dists)):
                if dists[i] - dists[i - 1] < 0:
                    count_moving_closer += 1
            obj_moving_closer[key] = count_moving_closer

        # returns obj that agent moved towards most of the time
        if len(obj_moving_closer) == 0:
            return goal
        max_key = max(obj_moving_closer.items(), key=operator.itemgetter(1))[0]
        # hack for making sure we're at least moving towards 1/2 time
        if obj_moving_closer[max_key] < len(objs) / 2:
            return goal
        goal[max_key] = objs[max_key]
        return goal

    def find_final_goal(self, images, d_maps):
        goals = []
        for i in range(0, len(images)):
            imgs, objs, agent_trajs = self.find_objs(images[i], d_maps[i])
            goal = self.find_goal(objs, agent_trajs)
            # stored as dict[trial number]=contour
            if bool(goal):
                key = list(goal.keys())[0]
                goals.append(goal[key][0])

        # finds the closest match goal
        final_goal = None
        cur_best_match = 1
        for i in range(1, len(goals)):
            match_score = cv2.matchShapes(goals[i], goals[i - 1], 1, 0.0)
            if match_score < cur_best_match:
                cur_best_match = match_score
                final_goal = goals[i]
        return final_goal

    def goal_in_image(self, goal, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask = self.remove_background(img)
        img, contours = self.find_contours(mask, img)
        best_match = 0.25
        xy = None
        for contour in contours:
            match_score = cv2.matchShapes(contours[contour][0], goal, 1, 0.0)
            if match_score < best_match:
                best_match = match_score
                xy = contour
        return xy, img

    def act(self, output):
        action = output.action_list[-1]
        # adds data by each trial
        self.trial = output.habituation_trial
        if self.trial is not None:
            if action == "Pass":
                self.trial_images[self.trial - 1].append(output.image_list[-1])
                self.trial_d_maps[self.trial - 1].append(output.depth_map_list[-1])
            return True
        else:
            agent_xy = None
            if action != "Pass":
                return True
            self.test_images.append(output.image_list[-1])
            self.test_d_maps.append(output.depth_map_list[-1])
            cur_frame = self.test_images[-1]
            # default no VOE heatmap
            empty_heatmap = self.test_d_maps[0] - self.test_d_maps[0]
            empty_heatmap_img = Image.fromarray(empty_heatmap)

            # if we haven't found the goal yet
            if self.goal_xy is None:
                # if we haven't found agent, cheat and use pink block
                if self.agent_xy is None:
                    agent_start = cv2.cvtColor(
                        np.array(cur_frame), cv2.COLOR_RGB2BGR)
                    lower_pink = np.array([180, 5, 180])
                    upper_pink = np.array([255, 150, 255])
                    pink_mask = cv2.inRange(
                        agent_start, lower_pink, upper_pink)
                    agent_start_img, agent_contour = self.find_contours(
                        pink_mask, agent_start)
                    # in case we don't find the pink block
                    if bool(agent_contour):
                        agent_xy = list(agent_contour.keys())[0]
                    else:
                        choice = "expected"
                        confidence = 1.0
                        violations_xy_list = [{}]
                        heatmap_img = empty_heatmap_img
                        return choice, confidence, violations_xy_list, heatmap_img

                # gets goal from habituation trials
                self.goal = self.find_final_goal(self.trial_images, self.trial_d_maps)
                # finds position of goal in test image
                self.xy, self.goal_img = self.goal_in_image(self.goal, cur_frame)
                # creates grid for A* star
                self.grid, temp_grid_img = self.create_grid(cur_frame, self.test_d_maps[-1])

                # obtains goal and agent locations
                if self.goal is not None and self.xy is not None:
                    self.goal_xy = [self.xy[0], self.xy[1]]
                if agent_xy is not None:
                    self.agent_xy = [agent_xy[0], agent_xy[1]]
                if self.goal_xy and self.agent_xy:
                    begin_dist = self.dist(self.goal_xy, self.agent_xy)

            # if we know goal, just need agent_xy from depth map change
            else:
                depth_diff = self.test_d_maps[-1] - self.test_d_maps[-2]
                heatmap_img = Image.fromarray(depth_diff, 'RGB')
                diff = np.nonzero(depth_diff)
                # if there's movement, we obtain new agent_xy
                if len(diff[0]) > 0:
                    agent_coord = self.get_agent_xy(diff[1], diff[0])
                    self.agent_xy = [agent_coord[0], agent_coord[1]]

            # if we still don't find goal or agent, return expected
            if self.agent_xy is None or self.goal_xy is None:
                choice = "expected"
                confidence = 0.0
                violations_xy_list = [{}]
                heatmap_img = empty_heatmap_img
                return choice, confidence, violations_xy_list, heatmap_img

            # otherwise, we can plan
            # creates plan using A* search
            if self.goal is not None and self.xy is not None:
                model = A_star_model(self.goal_xy, self.grid)
                planner = Astar_Planner(
                    model,
                    action_list,
                    np.Inf,
                    np.Inf,
                    goal_check,
                    heuristic,
                    state_to_tuple)
                plan = planner.plan(self.agent_xy, self.goal_xy)
                if plan is not None:
                    self.plan_lens.append(len(plan.state_traj))

            # also separately tracks difference array
            if len(self.plan_lens) > 1:
                self.plan_diff.append(self.plan_lens[-1] - self.plan_lens[-2])

            # makes frame by frame predictions
            # no VOE on first 3 frames
            if len(self.test_images) <= 3:
                if len(self.test_images) == 1 or not self.plan_diff:
                    choice = "expected"
                elif self.plan_diff[-1] <= 0:
                    choice = "expected"
                else:
                    choice = "unexpected"
                confidence = 0.5
                violations_xy_list = [{}]
                heatmap_img = empty_heatmap_img
                return choice, confidence, violations_xy_list, heatmap_img
            else:
                if self.plan_diff:
                    # average over last 3 steps
                    last_three_steps = self.plan_diff[-3:]
                    if np.average(last_three_steps) > 0:
                        choice = "unexpected"
                        last_three_bad_count = len(
                            [i for i in last_three_steps if i > 0])
                        confidence = last_three_bad_count / 3
                        violations_xy_list = [
                            {"x": float(self.agent_xy[0]), "y": float(self.agent_xy[1])}]
                        diff = self.test_d_maps[-1] - self.test_d_maps[-2]
                        diff[diff != 0] = 255
                        heatmap_img = Image.fromarray(diff)
                    else:
                        choice = "expected"
                        last_three_good_count = len(
                            [i for i in last_three_steps if i <= 0])
                        confidence = last_three_good_count / 3
                        violations_xy_list = [{}]
                        fake_diff = self.test_d_maps[-1] - self.test_d_maps[-1]
                        heatmap_img = Image.fromarray(fake_diff)

                    return choice, confidence, violations_xy_list, heatmap_img
                # error checks if we have no plan
                else:
                    choice = "expected"
                    confidence = 0.0
                    violations_xy_list = [{}]
                    heatmap_img = empty_heatmap_img
                    return choice, confidence, violations_xy_list, heatmap_img

    def get_trial_num(self):
        return self.trial

    def get_final_plausibility(self):
        if self.agent_xy and self.goal_xy:
            end_dist = self.dist(self.agent_xy, self.goal_xy)

        final_choice = "expected"
        final_confidence = 1.0

        # easiest is to look at last vs. first frame dist
        if self.begin_dist and self.end_dist:
            dist_traveled = self.end_dist - self.begin_dist
            # if we are closer than we started
            if dist_traveled < 0:
                final_choice = "expected"
                # obtains confidence from # of moves toward obj
                if self.plan_diff:
                    good_count = len([i for i in self.plan_diff if i <= 0])
                    final_confidence = float(good_count / len(self.plan_diff))
            else:
                final_choice = "unexpected"
                if self.plan_diff:
                    bad_count = len([i for i in self.plan_diff if i > 0])
                    final_confidence = float(bad_count / len(self.plan_diff))
        # otherwise, we need to look at planning traj
        else:            
            if self.plan_diff:
                avg_move = sum(self.plan_diff) / len(self.plan_diff)
                # unexpected if on average, moving away from goal
                if avg_move > 0:
                    final_choice = "unexpected"
                    bad_count = len([i for i in self.plan_diff if i > 0])
                    final_confidence = float(bad_count / len(self.plan_diff))
                # returns % of moves toward goal as confidence
                else:
                    final_choice = "expected"
                    good_count = len([i for i in self.plan_diff if i <= 0])
                    final_confidence = float(good_count / len(self.plan_diff))
            # we give up here
            else:
                final_choice = "expected"
                final_confidence = 0.0

        return final_choice, final_confidence