import numpy as np
from PIL import Image

def kernel_to_image(img_src_array, kernel, gap=None): 
    
    img_h,img_w = img_src_array.shape 
    ker_h,ker_w = kernel.shape 

    if gap==None:
        img_dst_array = img_src_array.copy()
    else:
        img_dst_array = np.zeros((img_h,img_w))
        img_dst_array.fill(gap)
    
    gap_area = ker_h//2 

    for h in range(gap_area,img_h-gap_area):
        for w in range(gap_area,img_w-gap_area):
            img_dst_array[h][w] = np.sum(img_src_array[h-gap_area:h+gap_area+1,w-gap_area:w+gap_area+1]*kernel)

    return img_dst_array

def NMS_to_image(G_norm_src, G_theta_src):
    img_h,img_w = G_norm_src.shape
    img_dist_array = G_norm_src.copy()
    G_norm_src[np.where((-22.5<=G_theta_src) & (G_theta_src<22.5))]=0
    G_norm_src[np.where((22.5<=G_theta_src) & (G_theta_src<67.5))]=45
    G_norm_src[np.where((67.5<=G_theta_src) & (G_theta_src<112.5))]=90
    G_norm_src[np.where((112.5<=G_theta_src) & (G_theta_src<157.5))]=135
    G_norm_src[np.where((157.5<=G_theta_src) & (G_theta_src<=180))]=0
    G_norm_src[np.where((-67.5<=G_theta_src) & (G_theta_src<-22.5))]=135
    G_norm_src[np.where((-112.5<=G_theta_src) & (G_theta_src<-67.5))]=90
    G_norm_src[np.where((-157.5<=G_theta_src) & (G_theta_src<-112.5))]=45
    G_norm_src[np.where((-180<=G_theta_src) & (G_theta_src<=-157.5))]=0

    for h in range(1,img_h-1):
        for w in range(1,img_w-1):
            if G_theta_src[h][w]==0:
                if G_norm_src[h][w]<G_norm_src[h][w-1] or G_norm_src[h][w]<G_norm_src[h][w+1]:
                    img_dist_array[h][w]=0
            if G_theta_src[h][w]==45:
                if G_norm_src[h][w]<G_norm_src[h-1][w+1] or G_norm_src[h][w]<G_norm_src[h+1][w-1]:
                    img_dist_array[h][w]=0
            if G_theta_src[h][w]==90:
                if G_norm_src[h][w]<G_norm_src[h-1][w] or G_norm_src[h][w]<G_norm_src[h+1][w]:
                    img_dist_array[h][w]=0
            if G_theta_src[h][w]==135:
                if G_norm_src[h][w]<G_norm_src[h-1][w-1] or G_norm_src[h][w]<G_norm_src[h+1][w+1]:
                    img_dist_array[h][w]=0
    print(img_dist_array.shape)
    return img_dist_array

def HT_to_image(src, HT_max, HT_min, gap_area=0): 
    img_h,img_w = src.shape # image height and width
    img_dist_array = src.copy() # make return image
    for h in range(gap_area, img_h-gap_area): # height
        for w in range(gap_area, img_w-gap_area): # width
            if src[h][w]>=HT_max: # if pix greater than HT_max
                img_dist_array[h][w]=255 # white
            elif src[h][w]<=HT_min: # if pix smaller than HT_min
                img_dist_array[h][w]=0 # black
            else:
                if np.max(src[h-gap_area:h+gap_area+1,w-gap_area:w+gap_area+1])>=HT_max: # if max neighbor pix greater than HT_max 
                    img_dist_array[h][w]=255 # white
                else: 
                    img_dist_array[h][w]=0 # black
    return img_dist_array

def threshold_otsu(src):
    histgram = np.array([np.sum(src==_) for _ in range(256)]) #horizon:luminance, vertical:quantity, the luminance pix quantity
    X_max = [-1,-1]
    for _ in range(1,256):
        n1 = np.sum(histgram[:_]) # sum from begin to luminance 
        n2 = np.sum(histgram[_:]) # sum from luminance to end
        s1 = np.var(histgram[:_]) # variance from begin to luminance
        s2 = np.var(histgram[_:]) # variance from luminance to end
        n_all=n1+n2 # sum from begin to end

        if n1==0: ave1=0 
        else:ave1 = np.sum(np.array([i*histgram[i] for i in range(0,_)]))/n1 # luminance ave from begin to luminance
        if n2==0: ave2=0
        else:ave2 = np.sum(np.array([i*histgram[i] for i in range(_,256)]))/n2 # luminance ave from luminance to end

        sb = (n1*n2/n_all**2)*((ave1-ave2)**2) # between class variance
        sw = (n1*s1/n_all)+(n2*s2/n_all) # within class variance
        X = sb/sw # good: sb is big, sw is small
        if X > X_max[1]: # if sb/sw is greater than before pix
            X_max = [_,X] # [luminance, sb/sw]
    
    threshold = X_max[0] # biggest sb and smallest sw luminance
    img_dist_array =  np.where(src>=threshold,255,0) 
    return img_dist_array

def threshold_adapt(src, kernel=3, th_gap=0): 
    k_gap = kernel//2
    src_h,src_w = src.shape # image height and width
    img_dist_array = np.empty((src_h,src_w)) # make return empty image
    N=kernel**2 

    for h in range(k_gap,src_h-k_gap): # height
        for w in range(k_gap,src_w-k_gap): # width
            ave = np.sum(src[h-k_gap:h+k_gap+1, w-k_gap:w+k_gap+1])/N # neighbor ave
            if src[h][w]<ave-th_gap:img_dist_array[h][w]=0 # if image less than ave, 0
            else: img_dist_array[h][w]=255 # else, 255
    return img_dist_array

def erode_to_image(src,kernel=3): 
    src_h,src_w=src.shape # image height and width
    img_dist_array = src.copy() # make return image
    k_gap = kernel//2 # kernel gap
    for h in range(k_gap,src_h-k_gap): # height
        for w in range(k_gap,src_w-k_gap): # width
            if np.any(src[h-k_gap:h+k_gap+1,w-k_gap:w+k_gap+1]==0.): # if any neighbor 0 
                img_dist_array[h][w]=0 # the pix 0
    return img_dist_array

def read_image(src):
    img_data    = Image.open(src).convert("L") 
    img_data    = np.array(src) 
    return img_data

def write_image(src,dist):
    img_output = Image.fromarray(src.astype(np.uint8)).convert("L")
    img_output.save(dist)

def return_kernel(name):
    if(name=="gaussian3"):
        return np.array([[1/16, 1/8, 1/16],
                         [1/8  ,1/4, 1/8],
                         [1/16, 1/8, 1/16]])
    elif(name=="gaussian5"):
        return np.array([[1,4,6,4,1],
                         [4,16,24,16,4],
                         [6,24,36,24,6],
                         [4,16,24,16,4],
                         [1,4,6,4,1]])/256
    elif(name=="sobel_x"):
        return np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1],])
    elif(name=="sobel_y"):
        return np.array([[-1,-2,-1],
                         [0,0,0],
                         [1,2,1]])
    elif(name=="laplacian"):
        return np.array([[1,1,1],
                         [1,-8,1],
                         [1,1,1]])
    else :
        return None
    
def norm_image(src_sobel_x,src_sobel_y):
    return np.sqrt(src_sobel_x**2 + src_sobel_y**2)

def theta_image(src_sobel_x,src_sobel_y):
    return np.arctan2(src_sobel_x,src_sobel_y)*(180/np.pi)