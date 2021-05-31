from math import sqrt
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
from cv2.cv2 import FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN


# 用于面部识别的类，能够根据输入的图片，输出结果
class FaceDetector(object):
    faceCascadePath = r'D:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
    eyeCascadePath = r'D:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml'
    readyToInit = True
    faceCascade = None
    eyeCascade = None
    img = None

    def __init__(self):
        pass

    def setPath(self, faceCascadePath, eyeCascadePath):
        try:
            self.faceCascadePath = faceCascadePath
            self.readyToInit = False
        except:
            print("faceCascadePath Error: Path not found.")
            self.readyToInit = False
            return False

        try:
            self.eyeCascadePath = eyeCascadePath
        except:
            print("eyeCascadePath Error: Path not found.")
            self.readyToInit = False
            return False

        self.readyToInit = True
        return True

    # 初始化 加载分类器
    def Init(self):
        if self.readyToInit:
            # 加载人脸人别分类器
            self.faceCascade = cv.CascadeClassifier(self.faceCascadePath)
            # 加载眼睛识别分类器
            self.eyeCascade = cv.CascadeClassifier(self.eyeCascadePath)
        else:
            print("Error: Not ready to init, set Path first")
            return False

    # 初始化之后输入一张RGB三通道图片
    def ReadImg(self, img):
        self.img = img
        return True

    # 处理读取的图片，返回人脸的坐标list和眼睛的坐标list
    def Detect(self):
        # 转换成灰度图像
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        # 人脸检测
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(32, 32)
        )

        # 在检测人脸的基础上检测眼睛
        eyes = []
        for (x, y, w, h) in faces:
            fac_gray = gray[y: (y + h), x: (x + w)]
            _eyes = self.eyeCascade.detectMultiScale(fac_gray, 1.3, 2)

            # 眼睛坐标的换算，将相对位置换成绝对位置
            for (ex, ey, ew, eh) in _eyes:
                eyes.append((x + ex, y + ey, ew, eh))

        return faces, eyes


# 获取两个点之间的距离的函数
# 输入深度本征，深度frame，两个点的像素坐标，输出的是两个点在三维空间的距离
# 原理是使用了 rs2_deproject_pixel_to_point() 函数，可以计算像素上某点对应的三维坐标，需要的参数是深度本征，像素坐标和深度。
def get_distance_between_points(intrinsics, depth_frame, point1_px, point1_py, point2_px, point2_py):
    point1_px = int(point1_px)
    point1_py = int(point1_py)
    point2_px = int(point2_px)
    point2_py = int(point2_py)
    dis1 = depth_frame.get_distance(point1_px, point1_py)
    dis2 = depth_frame.get_distance(point2_px, point2_py)
    point1_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, (point1_px, point1_py), dis1)
    point2_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, (point2_px, point2_py), dis2)
    # get distance 'cm'
    dist = sqrt(((point1_xyz[0] - point2_xyz[0]) ** 2) + ((point1_xyz[1] - point2_xyz[1]) ** 2) + (
            (point1_xyz[2] - point2_xyz[2]) ** 2)) * 100
    return dist


# 用于在图片上绘制两个点距离的函数
# 输入图像，距离，两个点的坐标，即可在它们之间绘制相应数据
# 和深度相机无关，只和opencv有关
def draw_distance_between_points(image, dist, point1_px, point1_py, point2_px, point2_py):
    point1_px = int(point1_px)
    point1_py = int(point1_py)
    point2_px = int(point2_px)
    point2_py = int(point2_py)
    circle_thickness = 4
    # draw circles
    cv.circle(image, (point1_px, point1_py), 6, (255, 0, 0), circle_thickness)
    cv.circle(image, (point2_px, point2_py), 6, (0, 255, 0), circle_thickness)
    # draw a line between them
    line_point_1 = (int(0.1 * point1_px + 0.9 * point2_px), int(0.1 * point1_py + 0.9 * point2_py))
    line_point_2 = (int(0.9 * point1_px + 0.1 * point2_px), int(0.9 * point1_py + 0.1 * point2_py))
    cv.line(image, line_point_1, line_point_2, (255, 255, 255), 4)
    # draw dist on line
    dist_text = str(round(dist, 3)) + "cm"
    cv.putText(image, dist_text,
               (int((line_point_1[0] + line_point_2[0]) / 2) - 30, int((line_point_2[0] + line_point_2[1]) / 2) + 30),
               FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def draw_faces(img, faces):
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


def draw_eyes(img, eyes):
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img


def draw_distance_between_faces(intrinsics, depth_frame_to_depth, img, faces):
    if len(faces) < 2:
        return img
    point1_x = faces[0][0] + faces[0][2] / 2
    point1_y = faces[0][1] + faces[0][3] / 2
    point2_x = faces[1][0] + faces[1][2] / 2
    point2_y = faces[1][1] + faces[1][3] / 2
    # 用自定义的函数获取这两个点的实际距离
    dist = get_distance_between_points(intrinsics, depth_frame_to_depth, point1_x, point1_y, point2_x, point2_y)
    # 用自定义函数将深度的信息绘制在图像上
    print(point1_x, point1_y, point2_x, point2_y)
    draw_distance_between_points(color_image_to_color, dist, point1_x, point1_y, point2_x, point2_y)
    return img

pipeline = rs.pipeline()

# 以下内容均为后处理，实际上暂时没什么用处
# 后处理1：定义一个深度着色器
color_map = rs.colorizer()
#   采取黑白着色
color_map_option_list = color_map.get_supported_options()
# print(option_list)
color_map.set_option(rs.option.color_scheme, 2.0)

# 后处理2：定义一个数据抽取器
dec = rs.decimation_filter()
dec_option_list = dec.get_supported_options()
# print(dec_option_list)
#   采取 2 的抽取程度
dec.set_option(rs.option.filter_magnitude, 2)

# 后处理3：定义一个深度到差距转换器 对D415这种结构光相机而言没有用处
depth2disparity = rs.disparity_transform()
depth2disparity_option_list = depth2disparity.get_supported_options()
# print(depth2disparity_option_list)
# 这个转换器没有设置的选项

# 后处理4：定义一个空间滤波器
spat = rs.spatial_filter()
spat_option_list = spat.get_supported_options()
# print(spat_option_list)
# 设置孔洞填充，这是一个不怎么好用的滤波器，但是暂时先忽略它的问题
spat.set_option(rs.option.holes_fill, 4)

# 后处理5：定义一个temporal filter (用于平滑)
temp = rs.temporal_filter()
temp_list = temp.get_supported_options()
# print(temp_list)
# 先备用
# temp.set_option()


# 设置对齐
# 设定需要对齐的方式（这里是深度对齐彩色，彩色图不变，深度图变换）
align_to_color = rs.stream.color
# 设定需要对齐的方式（这里是彩色对齐深度，深度图不变，彩色图变换）
align_to_depth = rs.stream.depth

alignedFs_to_color = rs.align(align_to_color)
alignedFs_to_depth = rs.align(align_to_depth)

# 设置config 用于控制pipeline
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# 启动pipeline
profile = pipeline.start(cfg)

# 减少孔洞的最好办法就是硬件上增大密度
# 可以直接在tool里面设置 可以看到开启高精度之后，孔洞多了很多


# opencv部分
faceDetector = FaceDetector()
faceDetector.Init()

# 开启摄像头
cap = cv.VideoCapture(1)
ok = True

try:
    while True:
        fs = pipeline.wait_for_frames()

        # 用两种方式进行对齐处理
        aligned_frames_to_color = alignedFs_to_color.process(fs)
        aligned_frames_to_depth = alignedFs_to_depth.process(fs)

        # 获取两种方法各自的深度frame和RGB frame 所以一共有四句代码
        color_frame_to_color = aligned_frames_to_color.get_color_frame()
        depth_frame_to_color = aligned_frames_to_color.get_depth_frame()

        color_frame_to_depth = aligned_frames_to_depth.get_color_frame()
        depth_frame_to_depth = aligned_frames_to_depth.get_depth_frame()

        # 对深度图调用后处理流程，spat暂时没用
        # color_frame_to_depth = spat.process(color_frame_to_depth)
        color_frame_to_depth = temp.process(color_frame_to_depth)
        # depth_frame_to_depth = spat.process(depth_frame_to_depth)
        depth_frame_to_depth = temp.process(depth_frame_to_depth)

        if not color_frame_to_color or not depth_frame_to_color or not color_frame_to_depth or not depth_frame_to_depth:
            continue

        # 将获取的四个frame分别用numpy的库函数转换为可以用opencv显示的image
        color_image_to_color = np.asanyarray(color_frame_to_color.get_data())
        depth_image_to_color = np.asanyarray(depth_frame_to_color.get_data())

        color_image_to_depth = np.asanyarray(color_frame_to_depth.get_data())
        depth_image_to_depth = np.asanyarray(depth_frame_to_depth.get_data())

        # 将深度图像染色，从而可以显示。实际使用的是对齐到深度的彩色图像，这里只是为了显示深度图，便于调试才进行的操作
        depth_image_to_color = cv.applyColorMap(cv.convertScaleAbs(depth_image_to_color, alpha=0.03), cv.COLORMAP_JET)
        depth_image_to_depth = cv.applyColorMap(cv.convertScaleAbs(depth_image_to_depth, alpha=0.03), cv.COLORMAP_JET)

        # 接下来的代码是根据需要的坐标获取其距离并绘制到图像上
        # 获取坐标

        # 读取摄像头中的RGB图像
        faceDetector.ReadImg(color_image_to_color)

        # 读取图像，返回脸和眼睛的坐标
        faces, eyes = faceDetector.Detect()

        # 绘制脸和眼睛
        color_image_to_color = draw_faces(color_image_to_color, faces)
        color_image_to_color = draw_eyes(color_image_to_color, eyes)

        # 获取深度本征，这和相机本身有关，和相机获取的帧无关
        stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        # 将depth_frame_to_depth转换为depth_frame类型，因为在之前的处理过程中，depth_frame_to_depth的类型变成了普通的frame类型
        depth_frame_to_depth = depth_frame_to_depth.as_depth_frame()
        # 绘制脸和脸之间的距离
        color_image_to_color = draw_distance_between_faces(intrinsics, depth_frame_to_depth, color_image_to_color, faces)

        # 拼接图像并显示，opencv的使用常识
        images_to_color = np.hstack((color_image_to_color, depth_image_to_color))
        images_to_depth = np.hstack((color_image_to_depth, depth_image_to_depth))

        cv.imshow('window_to_color', images_to_color)
        cv.imshow('window_to_depth', images_to_depth)

        cv.waitKey(1)
finally:
    pipeline.stop()
    cv.destroyAllWindows()
