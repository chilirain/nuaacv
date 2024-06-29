import cv2
import numpy as np
#1 获取暗通道图
def get_dark_channel(image, window_size=15):
    min_channel = np.min(image, 2)#对每个像素点取颜色通道的最小值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))#定义卷积核结构元素
    dark_channel = cv2.erode(min_channel, kernel)#进行最小值滤波
    return dark_channel#得到暗通道
#2 估计大气光A
def estimate_atmospheric_light(image, dark_channel):
    image_array=image.reshape(-1,3)
    dark_array=dark_channel.ravel()
    index=(-dark_array).argsort()[0]#取最亮的像素像素
    atmospheric_light=np.mean(image_array[index],0)
    return atmospheric_light
#3 估算传输图
#3.1 初步估算传输图t
def estimate_transmission(image, atmospheric_light, omega=0.95, window_size=15):
    norm_image=image/atmospheric_light#正则化
    dark_channel=get_dark_channel(norm_image,window_size)
    transmssion=1-omega*dark_channel
    return transmssion

#3.2 引导滤波优化传输图
def box_filter(img, r):
    """对图像进行盒滤波"""
    return cv2.blur(img, (2*r+1, 2*r+1))

def guided_filter(I, p, r, eps):
    """ 
    I: 引导图像（输入图像）
    p: 待滤波图像（透射率图）
    r: 半径
    eps: 正则化参数
    """ 
    q = np.zeros_like(p)
    I=I.astype(np.float32)
    p=p.astype(np.float32)
    for c in range(3):  # 对每个通道分别进行引导滤波
        Ic = I[:, :, c]
    #计算局部均值和相关性
        mean_I = box_filter(Ic, r)
        mean_p = box_filter(p, r)
        corr_Ip = box_filter(Ic * p, r)
        corr_II = box_filter(Ic * Ic, r)
    #计算方差和协方差
        var_I = corr_II - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
    #计算系数
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
    
        mean_a = box_filter(a, r)
        mean_b = box_filter(b, r)
        q += mean_a * Ic + mean_b
    q=q/3
    return q
#4 图像去雾
def get_recover_scene(image, transmission, atmospheric_light):
    transmission=np.maximum(transmission,0.1)#修正防止t太低
    transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    J=(image-atmospheric_light)/transmission+atmospheric_light
    J_clipped = np.clip(J, 0, 255).astype(np.uint8)
    return J_clipped

if __name__ == '__main__':
    image =cv2.imread('train.png')
    
dark_channel=get_dark_channel(image)   
atmospheric_light=estimate_atmospheric_light(image, dark_channel)   
initial_transmission=estimate_transmission(image,atmospheric_light)
r=80
eps=1e-3
transmission=guided_filter(image, initial_transmission, r, eps)
recover_scene=get_recover_scene(image, transmission, atmospheric_light)
cv2.imwrite('traind.png', dark_channel)
cv2.imwrite('traint.png', transmission)
#查看结果
cv2.imshow('Original Image', image)
cv2.imshow('Dark Channel', dark_channel)
cv2.imshow('Transmission', transmission)
cv2.imshow('Recovered Scene', recover_scene)
cv2.waitKey(0)
cv2.destroyAllWindows()