from threading import Thread
import time
from skimage import data, io
from matplotlib import pyplot as plt
import pyrealsense2 as rs
import numpy as np
from Commons import Display, Skipping
import cv2 as cv
import tensorflow as tf

class Core:
    def __init__(self, model):
        self.model = model

    def preproces_depth_image(self, raw_depth_frame):
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        depth_frame = spatial.process(raw_depth_frame)
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)
        depth_frame = temporal.process(depth_frame)
        hole_filling = rs.hole_filling_filter(2) # wypelnij wartoscia pixela ktory jest najblizej sensora
        depth_frame = hole_filling.process(depth_frame)
        return depth_frame

    def run_image(self, path_to_image, path_to_depth_image="", analyze_depth=False, plot = True):
        image = io.imread(path_to_image)
        depth_frame = []
        if path_to_depth_image:
            pipeline = rs.pipeline()
            config = rs.config()
            rs.config.enable_device_from_file(config, path_to_depth_image)
            pipeline.start(config)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            # depth_frame = self.preproces_depth_image(depth_frame)

        figsize = (13, 13)
        _, ax = plt.subplots(1, figsize=figsize)
        self.model.detect(image, ax, depth_frame, analyze_depth)
        if plot:
            plt.show()

    def run_frameset(self, path_to_frameset, display: Display = Display.plt, analyze_depth=True):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        rs.config.enable_device_from_file(config, path_to_frameset)
        colorizer = rs.colorizer()

        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align_to = rs.stream.color
        align = rs.align(align_to)

        figsize = (13, 13)
        _, ax = plt.subplots(1, figsize=figsize)
        if display is Display.plt:
            plt.ion()

        frames = pipeline.wait_for_frames(timeout_ms=200)

        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        self.model.detect(color_image, ax, depth_frame, analyze_depth)
        if display is Display.plt:
            plt.show()
            if not plt.get_fignums():
                return

    def run_video(self, path_to_video, skip_frames=Skipping.lost_frames, display: Display = Display.plt, analyze_depth=False):
        frames_to_skip = 5
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        rs.config.enable_device_from_file(config, path_to_video)

        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align_to = rs.stream.color
        align = rs.align(align_to)

        figsize = (13, 13)
        _, ax = plt.subplots(1, figsize=figsize)
        if display is Display.plt:
            plt.ion()
        is_first = True
        detection_thread = None
        it = 0

        while True:
            it = it + 1
            try:
                frames = pipeline.wait_for_frames(timeout_ms=200)
                if frames.size() < 2:
                    # Inputs are not ready yet
                    continue
            except (RuntimeError):
                pipeline.stop()
                break

            if skip_frames is Skipping.thread:
                if is_first is True:
                    is_first = False
                    detection_thread = Thread(target=self.run_detection, args=(frames, align, ax, display, analyze_depth))
                    detection_thread.start()
                    time.sleep(1)

                if not detection_thread.isAlive():
                    detection_thread.join()
                    detection_thread = Thread(target=self.run_detection, args=(frames, align, ax, display, analyze_depth))
                    detection_thread.start()

            elif skip_frames is Skipping.lost_frames:
                if it == frames_to_skip:
                    self.run_detection(frames, align, ax, display, analyze_depth)
                    it = 0
            else:
                self.run_detection(frames,align,ax,display,analyze_depth)


    def run_fast_ssd_video(self, path_to_video, display, analyze_depth):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        rs.config.enable_device_from_file(config, path_to_video)

        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        align_to = rs.stream.color
        align = rs.align(align_to)

        figsize = (13, 13)
        _, ax = plt.subplots(1, figsize=figsize)
        plt.ion()
        is_first = True
        detection_thread = None
        it = 0
        self.model.fast_detect(pipeline, align, analyze_depth)


    def run_detection(self, frames, align, ax, display, analyze_depth):
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        self.model.detect(color_image, ax, depth_frame, analyze_depth, display=display)

        if display is Display.plt:
            plt.pause(0.1)
            if not plt.get_fignums():
                exit(0)

        if display is Display.opencv:
            if cv.getWindowProperty("Detection result", cv.WND_PROP_VISIBLE) < 1:
                cv.destroyAllWindows()
                exit(0)


    def run_live(self, analyze_depth = False, display=Display.plt):
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

        # Start streaming

        ctx = rs.context()
        profile = ctx.devices[0]
        sensor_dep = profile.first_depth_sensor()
        sensor_color = profile.first_color_sensor()

        print("Exposure settings -> need to adjust to environment")
        sensor_dep.set_option(rs.option.enable_auto_exposure, 0)
        sensor_color.set_option(rs.option.enable_auto_exposure, 0)
        sensor_dep.set_option(rs.option.exposure, 800)
        sensor_color.set_option(rs.option.exposure, 600)

        print("exposure depth: %d" % sensor_dep.get_option(rs.option.exposure))
        print("exposure color: %d" % sensor_color.get_option(rs.option.exposure))
        depth_scale = sensor_dep.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        sensor_dep.set_option(rs.option.gain, 16)
        sensor_color.set_option(rs.option.brightness, 0)
        sensor_color.set_option(rs.option.contrast, 50)
        sensor_color.set_option(rs.option.hue, 0)
        sensor_color.set_option(rs.option.brightness, 0)
        sensor_color.set_option(rs.option.saturation, 64)
        sensor_color.set_option(rs.option.sharpness, 50)

        pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        figsize = (13, 13)
        _, ax = plt.subplots(1, figsize=figsize)
        if display is Display.plt:
            plt.ion()

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            raw_depth_frame = aligned_frames.get_depth_frame()
            raw_color_frame = aligned_frames.get_color_frame()

            depth_frame = self.preproces_depth_image(raw_depth_frame)
            color_image = np.asanyarray(raw_color_frame.get_data())
            color_image = cv.cvtColor(color_image, cv.COLOR_BGR2RGB)
            self.model.detect(color_image, ax, depth_frame, analyze_depth)
            if display is Display.plt:
                plt.pause(0.1)