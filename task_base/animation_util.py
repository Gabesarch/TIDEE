import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import utils.geom
from arguments import args
import torch
import os
import ipdb
st = ipdb.set_trace

class Animation():
    '''
    util for generating movies of the agent and TIDEE modules
    '''

    def __init__(self, W,H,navigation=None, name_to_id=None):  

        self.fig = plt.figure(1, dpi=args.dpi) 
        plt.clf()

        self.W = W
        self.H = H

        self.name_to_id = name_to_id

        self.object_tracker = None
        self.camX0_T_origin = None

        self.image_plots = []

        self.navigation = navigation

    def add_found_in_memory(self, obj_search, centroid):

        plt.figure(1, figsize=(14, 8)); plt.clf()

        m_vis = np.invert(self.navigation.explorer.mapper.get_traversible_map(
            self.navigation.explorer.selem, 1,loc_on_map_traversible=True))

        plt.imshow(m_vis, origin='lower', vmin=0, vmax=1,
                    cmap='Greys')

        # if self.object_tracker is not None:
        #     centroids, labels = self.object_tracker.get_centroids_and_labels()
        #     cmap = matplotlib.cm.get_cmap('gist_rainbow')
        #     obj_center_camX0 = utils.geom.apply_4x4(self.camX0_T_origin.float(), torch.from_numpy(centroids).unsqueeze(0).float()).squeeze().numpy()
        #     for o in range(len(obj_center_camX0)):
        #         label = labels[o]
        #         color_id = self.name_to_id[label]/len(self.name_to_id)
        #         color = cmap(color_id)
        #         obj_center_camX0_ = {'x':obj_center_camX0[o][0], 'y':obj_center_camX0[o][1], 'z':obj_center_camX0[o][2]}
        #         map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
        #         plt.plot(map_pos[0], map_pos[1], color=color, marker='o',linewidth=1, markersize=4)
        
        centroids, labels = self.object_tracker.get_centroids_and_labels()
        if not isinstance(centroids, int):
            if len(centroids)>0:
                if centroids.shape[1]>0:
                    cmap = matplotlib.cm.get_cmap('gist_rainbow')
                    obj_center_camX0 = centroids #utils.geom.apply_4x4(self.camX0_T_origin.float(), torch.from_numpy(centroids).unsqueeze(0).float()).squeeze(0).numpy()
                    for o in range(len(obj_center_camX0)):
                        label = labels[o]
                        if label not in self.name_to_id.keys():
                            continue
                        color_id = self.name_to_id[label]/len(self.name_to_id)
                        color = cmap(color_id)
                        obj_center_camX0_ = {'x':obj_center_camX0[o][0], 'y':obj_center_camX0[o][1], 'z':obj_center_camX0[o][2]}
                        map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
                        plt.plot(map_pos[1], map_pos[0], color=color, marker='o',linewidth=1, markersize=4)

        # plt.set_title("Semantic Map")

        plt.xticks([])
        plt.yticks([])
        plt.gca().axis('off')


        plt.title(f'Receptacle {obj_search} found in memory',fontsize= 20)
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        semantic_map = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        obj_center_camX0 = centroid #utils.geom.apply_4x4(self.camX0_T_origin.float(), torch.from_numpy(centroid).unsqueeze(0).unsqueeze(0).float()).squeeze().numpy()
        obj_center_camX0 = {'x':obj_center_camX0[0], 'y':obj_center_camX0[1], 'z':obj_center_camX0[2]}
        map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0)
        color_id = self.name_to_id[obj_search]/len(self.name_to_id)
        color = cmap(color_id)
        plt.plot(map_pos[1], map_pos[0], color=color, marker='o',linewidth=1, markersize=8, markeredgewidth=1, markeredgecolor='black')

        plt.title(f'Receptacle {obj_search} found in memory',fontsize= 20)
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        semantic_map2 = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        for i in range(10):
            self.image_plots.append(semantic_map)
            self.image_plots.append(semantic_map2)


    def add_active_search_visual(self, furthest_pts, scores_argsort, feat_mem_logits_vis, thresh_mem, obj_search, add_map=True):
        
        plt.figure(1, figsize=(14, 8)); plt.clf()
        # plt.imshow(rgb)
        
        # plt.title(name)
        # plt.savefig(os.path.join(root_folder, f'{key}-{receptacle}.png'), bbox_inches='tight')
        # plt.subplot(1,2,2)
        m_vis = np.invert(self.navigation.explorer.mapper.get_traversible_map(
            self.navigation.explorer.selem, 1,loc_on_map_traversible=True))

        plt.imshow(m_vis, origin='lower', vmin=0, vmax=1,
                    cmap='Greys')
        # state_xy = self.navigation.explorer.mapper.get_position_on_map()
        # state_theta = self.navigation.explorer.mapper.get_rotation_on_map()
        # arrow_len = 2.0/self.navigation.explorer.mapper.resolution
        # plt.arrow(state_xy[0], state_xy[1], 
        #             arrow_len*np.cos(state_theta+np.pi/2),
        #             arrow_len*np.sin(state_theta+np.pi/2), 
        #             color='b', head_width=20)
        
        # with open('images/m_vis.npy', 'wb') as f:
        #     np.save(f, m_vis)
        # with open('images/state_xy.npy', 'wb') as f:
        #     np.save(f, state_xy)
        # with open('images/state_theta.npy', 'wb') as f:
        #     np.save(f, state_theta)
        # with open('images/arrow_len.npy', 'wb') as f:
        #     np.save(f, arrow_len)

        # if self.navigation.explorer.point_goal is not None:
        #     ax[0].plot(self.navigation.explorer.point_goal[1], self.navigation.explorer.point_goal[0], color='blue', marker='o',linewidth=10, markersize=12)
        #ax[2].set_title(f"Traversable {self.unexplored_area}")

        if self.object_tracker is not None:
            # centroids, labels = self.object_tracker.get_centroids_and_labels()
            # cmap = matplotlib.cm.get_cmap('gist_rainbow')
            # obj_center_camX0 = utils.geom.apply_4x4(self.camX0_T_origin.float(), torch.from_numpy(centroids).unsqueeze(0).float()).squeeze().numpy()
            # for o in range(len(obj_center_camX0)):
            #     label = labels[o]
            #     color_id = self.name_to_id[label]/len(self.name_to_id)
            #     color = cmap(color_id)
            #     obj_center_camX0_ = {'x':obj_center_camX0[o][0], 'y':obj_center_camX0[o][1], 'z':obj_center_camX0[o][2]}
            #     map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
            #     plt.plot(map_pos[0], map_pos[1], color=color, marker='o',linewidth=1, markersize=4)

            centroids, labels = self.object_tracker.get_centroids_and_labels()
            if not isinstance(centroids, int):
                if len(centroids)>0:
                    if centroids.shape[1]>0:
                        cmap = matplotlib.cm.get_cmap('gist_rainbow')
                        obj_center_camX0 = centroids #utils.geom.apply_4x4(self.camX0_T_origin.float(), torch.from_numpy(centroids).unsqueeze(0).float()).squeeze(0).numpy()
                        for o in range(len(obj_center_camX0)):
                            label = labels[o]
                            if label not in self.name_to_id.keys():
                                continue
                            color_id = self.name_to_id[label]/len(self.name_to_id)
                            color = cmap(color_id)
                            obj_center_camX0_ = {'x':obj_center_camX0[o][0], 'y':obj_center_camX0[o][1], 'z':obj_center_camX0[o][2]}
                            map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
                            plt.plot(map_pos[1], map_pos[0], color=color, marker='o',linewidth=1, markersize=4)
                            # ax[1].plot(map_pos[1], map_pos[0], color=color, marker='o',linewidth=1, markersize=4)

        # plt.set_title("Semantic Map")

        plt.xticks([])
        plt.yticks([])
        plt.gca().axis('off')

        
        # ax = plt.gca()

        # ax.text(0.0,0.0,"Test", fontsize=45)
        # ax.axis('off')
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        semantic_map = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)


        ##############
        plt.figure(1, figsize=(14, 8)); plt.clf()
        plt.imshow(feat_mem_logits_vis, origin='lower', vmin=0, vmax=1,
                    cmap='RdBu')
        plt.xticks([])
        plt.yticks([])
        plt.gca().axis('off')
        # plt.title(f'Where to search for {obj_search}?')
        plt.xlabel('sigmoid heatmap output')
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        heatmap = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        ##########
        plt.figure(1, figsize=(14, 8)); plt.clf()
        plt.imshow(thresh_mem, origin='lower', vmin=0, vmax=1,
                    cmap='RdBu')
        plt.xticks([])
        plt.yticks([])
        plt.gca().axis('off')
        # plt.title(f'Where to search for {obj_search}?')
        plt.title('threshold heatmap')
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        thresh = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # plt.title(f'Where to search for {obj_search}?')
        plt.title('search locations - farthest point sampling')
        plt.scatter(furthest_pts[scores_argsort[0:1],1], furthest_pts[scores_argsort[0:1],0], color='lime', s=15)
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        thresh_fps1 = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # plt.title(f'Where to search for {obj_search}?')
        plt.title('search locations - farthest point sampling')
        plt.scatter(furthest_pts[scores_argsort[0:2],1], furthest_pts[scores_argsort[0:2],0], color='lime', s=15)
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        thresh_fps2 = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # plt.title(f'Where to search for {obj_search}?')
        plt.title('search locations - farthest point sampling')
        plt.scatter(furthest_pts[scores_argsort[0:3],1], furthest_pts[scores_argsort[0:3],0], color='lime', s=15)
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()       
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        thresh_fps3 = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # plt.title('farthest point sampling')
        # plt.scatter(furthest_pts[scores_argsort[0:4],1], furthest_pts[scores_argsort[0:4],0])
        # canvas = FigureCanvas(plt.gcf())
        # canvas.draw()       
        # width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        # thresh_fps4 = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # plt.title('farthest point sampling')
        # plt.scatter(furthest_pts[scores_argsort[0:5],1], furthest_pts[scores_argsort[0:5],0])
        # canvas = FigureCanvas(plt.gcf())
        # canvas.draw()       
        # width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        # thresh_fps5 = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        #############
        text_size=0.7; text_th=1
        root = '/home/gsarch/repo/project_cleanup/phase_base/animation_images/'
        as1 = cv2.imread(os.path.join(root, 'activesearch1.png'), flags=cv2.IMREAD_COLOR)
        as2 = cv2.imread(os.path.join(root, 'activesearch2.png'), flags=cv2.IMREAD_COLOR)
        as3 = cv2.imread(os.path.join(root, 'activesearch3.png'), flags=cv2.IMREAD_COLOR)

        ###########
        # put it all together
        semantic_map2 = semantic_map[:,240:480]

        W = semantic_map2.shape[1]
        pad_am = 640-W
        combo_image = np.concatenate([semantic_map2, 255*np.ones((480,pad_am,3)).astype(np.uint8)], axis=1)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)

        combo_image = np.concatenate([semantic_map2, as1[:,240:]], axis=1)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)

        combo_image = np.concatenate([semantic_map2, as2[:,240:]], axis=1)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)

        cv2.putText(combo_image,obj_search,(int(365), int(365)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)

        combo_image = np.concatenate([semantic_map2, as3[:,240:]], axis=1)
        cv2.putText(combo_image,obj_search,(int(365), int(365)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)


        # cv2.imwrite('images/test.png', combo_image)

        
        

        # cv2.imwrite('images/test.png', heatmap)
        heatmap2 = heatmap[:,140:540]
        combo_image = np.concatenate([semantic_map2, heatmap2], axis=1)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)

        thresh2 = thresh[:,140:540]
        combo_image = np.concatenate([semantic_map2, thresh2], axis=1)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)

        thresh2 = thresh_fps1[:,140:540]
        combo_image = np.concatenate([semantic_map2, thresh2], axis=1)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)

        thresh2 = thresh_fps2[:,140:540]
        combo_image = np.concatenate([semantic_map2, thresh2], axis=1)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)

        thresh2 = thresh_fps3[:,140:540]
        combo_image = np.concatenate([semantic_map2, thresh2], axis=1)
        cv2.putText(combo_image,f'Where to search for the {obj_search}?',(int(150), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        for i in range(5):
            self.image_plots.append(combo_image)


        



        # cv2.imwrite('images/test.png', combo_image)

        

        # cv2.imwrite('images/test.png', thresh)



        # st()

        # cv2.imwrite('images/test.png', image)
        # plt.figure()
        # plt.imshow(image)
        # plt.savefig('images/test.png')
        # st()

    def add_rgcn_visual(self, objectType, top_k_classes_b):
        text_size=0.7; text_th=1
        root = '/home/gsarch/repo/project_cleanup/phase_base/animation_images/'
        rgcn1 = cv2.imread(os.path.join(root, 'rGCNvisual1.png'), flags=cv2.IMREAD_COLOR)
        rgcn2 = cv2.imread(os.path.join(root, 'rGCNvisual2.png'), flags=cv2.IMREAD_COLOR)
        rgcn3 = cv2.imread(os.path.join(root, 'rGCNvisual3.png'), flags=cv2.IMREAD_COLOR)
        rgcn4 = cv2.imread(os.path.join(root, 'rGCNvisual4.png'), flags=cv2.IMREAD_COLOR)
        rgcn5 = cv2.imread(os.path.join(root, 'rGCNvisual5.png'), flags=cv2.IMREAD_COLOR)

        rgcn1 = np.float32(rgcn1)
        rgcn2 = np.float32(rgcn2)
        rgcn3 = np.float32(rgcn3)
        rgcn4 = np.float32(rgcn4)
        rgcn5 = np.float32(rgcn5)
        cv2.putText(rgcn1,objectType,(int(30), int(242)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)
        cv2.putText(rgcn2,objectType,(int(30), int(242)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)
        cv2.putText(rgcn3,objectType,(int(30), int(242)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)
        cv2.putText(rgcn4,objectType,(int(30), int(242)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)
        cv2.putText(rgcn5,objectType,(int(30), int(242)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)
        rgcn6 = rgcn5.copy()
        starts = [206+10,220+10,236+10,250+10,265+10] 
        for cl in range(len(top_k_classes_b)):
            cv2.putText(rgcn6,top_k_classes_b[cl],(int(472), int(starts[cl])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),thickness=1)
        rgcn7 = rgcn5.copy()
        for cl in range(len(top_k_classes_b)):
            if cl==0:
                thick = 2
            else:
                thick = 1
            cv2.putText(rgcn7,top_k_classes_b[cl],(int(472), int(starts[cl])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),thickness=thick)

        for i in range(5):
            self.image_plots.append(rgcn1)
        for i in range(5):
            self.image_plots.append(rgcn2)
        for i in range(5):
            self.image_plots.append(rgcn3)
        for i in range(5):
            self.image_plots.append(rgcn4)
        for i in range(5):
            self.image_plots.append(rgcn5)
        for i in range(5):
            self.image_plots.append(rgcn6)
            self.image_plots.append(rgcn7)

        # st()

        # cv2.imwrite('images/test.png', rgcn6)

        # plt.figure()
        # plt.imshow(rgcn6)
        # plt.savefig('images/test.png')



    def add_text_only(self, text=None, add_map=True):

        image = np.ones((self.W,self.H, 3))*255.

        ncols = 2
        plt.clf()
        
        ax = []
        spec = gridspec.GridSpec(ncols=ncols, nrows=1, 
                figure=self.fig, left=0., right=1., wspace=0.05, hspace=0.5)
        ax.append(self.fig.add_subplot(spec[0, 0]))
        if add_map:
            ax.append(self.fig.add_subplot(spec[0, 1]))

        for a in ax:
            a.axis('off')

        t_i = 1
        for t_ in text:
            text_size=0.5; text_th=1
            image = np.float32(image)
            cv2.putText(np.float32(image),t_,(int(20), int(20*t_i)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)
            t_i += 1

        ax[0].set_title("")

        image = image.astype(np.uint8)
        # plt.subplot(1,2,1)
        ax[0].imshow(image)

        if add_map and self.navigation is not None:
            # plt.subplot(1,2,2)
            m_vis = np.invert(self.navigation.explorer.mapper.get_traversible_map(
                self.navigation.explorer.selem, 1,loc_on_map_traversible=True))

            ax[1].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                     cmap='Greys')
            state_xy = self.navigation.explorer.mapper.get_position_on_map()
            state_theta = self.navigation.explorer.mapper.get_rotation_on_map()
            arrow_len = (2.0/self.navigation.explorer.mapper.resolution)/2.0

            ax[1].arrow(state_xy[0], state_xy[1], 
                        arrow_len*np.cos(state_theta+np.pi/2),
                        arrow_len*np.sin(state_theta+np.pi/2), 
                        color='b', head_width=20)
            if self.navigation.explorer.point_goal is not None:
                ax[1].plot(self.navigation.explorer.point_goal[1], self.navigation.explorer.point_goal[0], color='blue', marker='o',linewidth=10, markersize=12)
            #ax[2].set_title(f"Traversable {self.unexplored_area}")
            ax[1].set_title("Obstacle Map")

        canvas = FigureCanvas(plt.gcf())
        # ax = plt.gca()

        # ax.text(0.0,0.0,"Test", fontsize=45)
        # ax.axis('off')

        canvas.draw()       # draw the canvas, cache the renderer
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # plot_to_add = plt.gcf()
        # self.image_plots.append([plot_to_add])

        self.image_plots.append(image)



    def add_frame(self, image, text=None, add_map=True, box=None):
        # print('here')

        # if add_map:
        #     ncols = 2
        # else:
        #     ncols = 1

        ncols = 2
        plt.figure(1)
        plt.clf()
        
        ax = []
        spec = gridspec.GridSpec(ncols=ncols, nrows=1, 
                figure=self.fig, left=0., right=1., wspace=0.05, hspace=0.5)
        ax.append(self.fig.add_subplot(spec[0, 0]))
        if add_map:
            ax.append(self.fig.add_subplot(spec[0, 1]))

        for a in ax:
            a.axis('off')

        if text is not None:
            text_size=0.5; text_th=1
            image = np.float32(image)
            # cv2.putText(image,text,(int(20), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
            ax[0].set_title(text)

        if box is not None:
            rect_th=1; text_size=text_size; text_th=1
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0, 255, 0), rect_th)
            # st()
            # plt.figure()
            # plt.imshow(image.astype(np.int32))
            # plt.savefig('images/test.png')

        image = image.astype(np.uint8)
        # plt.subplot(1,2,1)
        ax[0].imshow(image)
        
        

        if add_map and self.navigation is not None:
            # plt.subplot(1,2,2)
            m_vis = np.invert(self.navigation.explorer.mapper.get_traversible_map(
                self.navigation.explorer.selem, 1,loc_on_map_traversible=True))

            ax[1].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                     cmap='Greys')
            state_xy = self.navigation.explorer.mapper.get_position_on_map()
            state_theta = self.navigation.explorer.mapper.get_rotation_on_map()
            arrow_len = 2.0/self.navigation.explorer.mapper.resolution
            ax[1].arrow(state_xy[0], state_xy[1], 
                        arrow_len*np.cos(state_theta+np.pi/2),
                        arrow_len*np.sin(state_theta+np.pi/2), 
                        color='b', head_width=20)
            
            # with open('images/m_vis.npy', 'wb') as f:
            #     np.save(f, m_vis)
            # with open('images/state_xy.npy', 'wb') as f:
            #     np.save(f, state_xy)
            # with open('images/state_theta.npy', 'wb') as f:
            #     np.save(f, state_theta)
            # with open('images/arrow_len.npy', 'wb') as f:
            #     np.save(f, arrow_len)

            if self.navigation.explorer.point_goal is not None:
                ax[1].plot(self.navigation.explorer.point_goal[1], self.navigation.explorer.point_goal[0], color='blue', marker='o',linewidth=10, markersize=12)
            #ax[2].set_title(f"Traversable {self.unexplored_area}")

            if self.object_tracker is not None:
                centroids, labels = self.object_tracker.get_centroids_and_labels()
                if not isinstance(centroids, int):
                    if len(centroids)>0:
                        if centroids.shape[1]>0:
                            cmap = matplotlib.cm.get_cmap('gist_rainbow')
                            obj_center_camX0 = centroids #utils.geom.apply_4x4(self.camX0_T_origin.float(), torch.from_numpy(centroids).unsqueeze(0).float()).squeeze(0).numpy()
                            for o in range(len(obj_center_camX0)):
                                label = labels[o]
                                if label not in self.name_to_id.keys():
                                    continue
                                color_id = self.name_to_id[label]/len(self.name_to_id)
                                color = cmap(color_id)
                                obj_center_camX0_ = {'x':obj_center_camX0[o][0], 'y':obj_center_camX0[o][1], 'z':obj_center_camX0[o][2]}
                                map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
                                ax[1].plot(map_pos[1], map_pos[0], color=color, marker='o',linewidth=1, markersize=4)



            ax[1].set_title("Semantic Map")

        canvas = FigureCanvas(plt.gcf())
        # ax = plt.gca()

        # ax.text(0.0,0.0,"Test", fontsize=45)
        # ax.axis('off')

        canvas.draw()       # draw the canvas, cache the renderer
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        self.image_plots.append(image)

        # # put pixel buffer in numpy array
        # canvas = FigureCanvas(fig)
        # canvas.draw()
        # mat = np.array(canvas.renderer._renderer)
        # mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

    def render_movie(self, dir,  episode, tag='', fps=5):

        # rand_int = np.random.randint(100)

        if not os.path.exists(dir):
            os.mkdir(dir)
        video_name = os.path.join(dir, f'output{episode}_{tag}.mp4')
        print(f"rendering to {video_name}")
        height, width, _ = self.image_plots[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 4, (width, height))
        # cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))

        for im in self.image_plots:
            rgb = np.array(im).astype(np.uint8)
            bgr = rgb[:,:,[2,1,0]]
            video_writer.write(bgr)

        cv2.destroyAllWindows()
        video_writer.release()



    