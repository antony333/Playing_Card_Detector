import streamlit as st
import cv2
import numpy as np
from PIL import Image



#Code for streamlit
st.markdown("<h1 style='text-align: center; font-size: 40px;'>Playing Card Detector</h1>",unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 20px;'>Team Members : Antony Jerald & Joel Joy</h1>", unsafe_allow_html=True)


#To coose Image
input_image = st.radio("Choose if you want to work on already given dataset or use your own image",('Given Iamge', 'Your Image'))
image = []


if input_image == 'Given Iamge':
  Data_Set = st.selectbox('Select Data Set:',('FourHearts','ThreeClubs','SixDiamonds','TwoDiamonds','JHearts'))
  st.write('Selected Data is:', Data_Set)

  if(Data_Set == "FourHearts"):
    image = open("./Test_Images/4Hearts.jpeg", "rb").read()
    st.image(image, caption='Selected Card for detection',width = 400)

  if(Data_Set == "ThreeClubs"):
    image = open("./Test_Images/3Clubs.jpeg", "rb").read()
    st.image(image, caption='Selected Card for detection',width = 400)

  if(Data_Set == "SixDiamonds"):
    image = open("./Test_Images/6Diamonds.jpeg", "rb").read()
    st.image(image, caption='Selected Card for detection',width = 400)

  if(Data_Set == "TwoDiamonds"):
    image = open("./Test_Images/2Diamonds.jpeg", "rb").read()
    st.image(image, caption='Selected Card for detection',width = 400)
  
  if(Data_Set == "JHearts"):
    image = open("./Test_Images/JHearts.jpeg", "rb").read()
    st.image(image, caption='Selected Card for detection',width = 400)
	
  if(Data_Set == "FourHearts"):
      image = cv2.imread(r'./Test_Images/4Hearts.jpeg')
  elif(Data_Set == "ThreeClubs"):
      image = cv2.imread(r'./Test_Images/3Clubs.jpeg')
  elif(Data_Set == "TwoDiamonds"):
      image = cv2.imread(r'./Test_Images/2Diamonds.jpeg')
  elif(Data_Set == "SixDiamonds"):
      image = cv2.imread(r'./Test_Images/6Diamonds.jpeg')
  elif(Data_Set == "JHearts"):
      image = cv2.imread(r'./Test_Images/JHearts.jpeg')



#To give our own input image
else:	
  uploaded_file = st.file_uploader("Upload Card Iamge (For better result use images in dark background)", type=['jpeg', 'png', 'jpg'])

  if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR")




#Main Code Starts

#%%  Some initialisation
#Maximum and Minimum Card Areas
if (len(image)!=0):
    Min_Card_Area = 25000
    Max_Card_Area = 300000
    # Adaptive threshold levels
    const_for_thresh = 100
    # Height and width of rectangle which contains card suit aand rank
    Corner_H = 84
    Corner_W = 32
    # Dimensions of suit train images
    Suit_H = 100
    Suit_W = 70
    # Dimensions of rank train images
    Rank_H = 125
    Rank_W = 70
    # Maximum difference in rank allowed
    Max_Suit_Diff = 1100
    Max_Rank_Diff = 3500

    
    class Train_ranks:
        def __init__(self):
            self.img = []
            self.name = "Dummy"
    
    
    class Train_suits:
        def __init__(self):
            self.img = []
            self.name = "Dummy"
    
    def load_ranks(filepath):    
        train_ranks = []
        i = 0    
        for Rank in ['Ace','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten','Jack','Queen','King']:
            train_ranks.append(Train_ranks())
            train_ranks[i].name = Rank
            filename = Rank + '.jpg'
            train_ranks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
            i = i + 1
    
        return train_ranks
    
    
    def load_suits(filepath):
        train_suits = []
        i = 0   
        for Suit in ['Spades','Diamonds','Clubs','Hearts']:
            train_suits.append(Train_suits())
            train_suits[i].name = Suit
            filename = Suit + '.jpg'
            train_suits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
            i = i + 1
    
        return train_suits
    
    
    
    #train_ranks = load_ranks( path + '/Card_Images/')
    #train_suits = load_suits( path + '/Card_Images/')
    
    train_ranks = load_ranks('Card_Images/')
    train_suits = load_suits('Card_Images/')
    
    
    #%% Preprocessing Part
    
    
    #To convert to gray scale and blur the image to remove noise if present
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img,(3,3),0)
    
    
    #To threshold the card alone from the image
    #For this we consider a sample pixel loacated at the middle of the image and at top
    #We use this as the background pixel and add a constant value THRESH_ADDER 
    #such that pixels above that value will be thresholded
    
    img_w, img_h = np.shape(image)[:2]
    bkg = gray_img[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg + const_for_thresh
    _, thresh_image = cv2.threshold(blur_img,thresh_level,255,cv2.THRESH_BINARY)
    
    
    
    
    #%% To find contours in the image
    
    
    #The hierarchy array can be used to determine whether or not the contours have parents.
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #To sort the countours in the decreasing order of size 
    ind_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
    
    
    contour_sorted = []
    hierarchy_sorted = []
    contour_is_card = np.zeros(len(contours),dtype=int)
    
        
        
    # To get an array of countours and heirarchy with decreasing order of size    
    for i in ind_sort:
        contour_sorted.append(contours[i])
        hierarchy_sorted.append(hierarchy[0][i])   
    

    
    #%% Card Iamge Processing
    
    
    for i in range(len(contour_sorted)):
        size = cv2.contourArea(contour_sorted[i])
        perimeter = cv2.arcLength(contour_sorted[i],True)
        approx = cv2.approxPolyDP(contour_sorted[i],0.01*perimeter,True)
        
        
        #We can then apply the following criteria to determine whether contours are cards: 
        #1) Have four corners
        #2) Smaller than the maximum card size
        #3) larger than the minimum card size
        #4) do not have parents
        if ((len(approx) == 4) and (size < Max_Card_Area) and (size > Min_Card_Area) and (hierarchy_sorted[i][3] == -1) ):
            contour_is_card[i] = 1
    
    
        
    if len(contour_sorted) == 0:
        print("No card found in the image")    
    else:    
        
    
    #To take the card shape and finds the attributes of the card (corner points, etc). 
    #To create a flattened 200x300 image of the card and extracts the suit and rank 
    #of the card from the image.
    
        cards = []
        for i in range(len(contour_sorted)):
            if (contour_is_card[i] == 1):
                contour = contour_sorted[i]
                # To use the perimeter of the contour to approximate the corner points
                peri = cv2.arcLength(contour,True)
                approx = cv2.approxPolyDP(contour,0.01*peri,True)
                pts = np.float32(approx)
                corner_pts = pts
                
                # To determine the card's bounding rectangle's height and breadth
                x,y,w,h = cv2.boundingRect(contour)
                width, height = w, h
            
                #To calculate the card's centre point, take the average of the four corner points.
                average = np.sum(pts, axis=0)/len(pts)
                cent_x = int(average[0][0])
                cent_y = int(average[0][1])
                center = [cent_x, cent_y]
                
                
                
                
    #%%         To Warp the card into a 200x300 flattened image using the perspective transform.
                
    
    
                #To create a list of coordinates that will be sorted so that the 
                #first item is top-left, the second is top-right, 
                #the third is bottom-right, and the fourth is bottom-left.
                rect = np.zeros((4,2), dtype = "float32")
                
            	
                # bottom-left and top-right point will have the largest and smallest difference respectively
                diff = np.diff(pts, axis = -1)
                topright = pts[np.argmin(diff)]
                bottomleft = pts[np.argmax(diff)]
                
                
                # bottom-right and top-left point will have the largest and smallest sum respectively
                s = np.sum(pts, axis = 2)
                topleft = pts[np.argmin(s)]
                bottomright = pts[np.argmax(s)]
            
                
                #Case when card is horizontally oriented
                if w >= 1.2*h: 
                    rect[0] = bottomleft
                    rect[1] = topleft
                    rect[2] = topright
                    rect[3] = bottomright
                
                
                # Case when card is vertically oriented
                if w <= 0.8*h: 
                    rect[0] = topleft
                    rect[1] = topright
                    rect[2] = bottomright
                    rect[3] = bottomleft
            
                
                
                # Case when card is oriented in diamond shape
                if w > 0.8*h and w < 1.2*h:
                # If furthest left point is higher than furthest right point then card is tilted to the left.
             
                    if pts[1][0][1] <= pts[3][0][1]:
                    # cv2.approxPolyDP returns points in the following sequence if the card is titled to the left:
                    #top right, top left, bottom left, bottom right.
                        rect[0] = pts[1][0] 
                        rect[1] = pts[0][0]
                        rect[2] = pts[3][0]
                        rect[3] = pts[2][0]
            
                    # If furthest left point is lower than furthest right point, card is tilted to the right
                    if pts[1][0][1] > pts[3][0][1]:
                        # cv2.approxPolyDP returns points in the sequence if the card is titled to the left:
                        #top left, bottom left, bottom right, top right
                        rect[0] = pts[0][0] 
                        rect[1] = pts[3][0] 
                        rect[2] = pts[2][0] 
                        rect[3] = pts[1][0] 
                        
                    
                maxWidth = 200
                maxHeight = 300
            
                # To Make a destination array, a perspective transform matrix, and a warp card image
                dest = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
                Matrix = cv2.getPerspectiveTransform(rect,dest)
                warp = cv2.warpPerspective(image, Matrix, (maxWidth, maxHeight))
                warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
            
    
    
                
    #%%            
    
                # to Grab a corner of the warped card image and zoom in 4 times.
                #RScorner means RankSuitcorner
                RScorner = warp[0:Corner_H, 0:Corner_W]
                RScorner_zoom = cv2.resize(RScorner, (0,0), fx=4, fy=4)
                
                
                #Using the threshold level as 155 to binarise the image
                thresh_level = 155
                _, Fcard_thresh = cv2.threshold(RScorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
                
                #Dividing the obtained rank-suit image into two halves Top half : rank and bottom half suit
                #Frank means final rank
                Frank = Fcard_thresh[20:185, 0:128]
                Fsuit = Fcard_thresh[186:336, 0:128]
                
                
                
                #To Isolate and identify the biggest contour to discover the bounding rectangle and rank contour.
                Frank_cnts, hier = cv2.findContours(Frank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                Frank_cnts = sorted(Frank_cnts, key=cv2.contourArea,reverse=True)
            
                
                #To determine the bounding rectangle of the largest contour and 
                #use it to scale the acquired rank image to match the size of the train rank image.
                if len(Frank_cnts) != 0:
                    x1,y1,w1,h1 = cv2.boundingRect(Frank_cnts[0])
                    Frank_roi = Frank[y1:y1+h1, x1:x1+w1]
                    Frank_sized = cv2.resize(Frank_roi, (Rank_W,Rank_H), 0, 0)
                    rank_img = Frank_sized
                
                
                #To Isolate and identify the biggest contour to determine the bounding rectangle and suit contour.
                Fsuit_cnts, hier = cv2.findContours(Fsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                Fsuit_cnts = sorted(Fsuit_cnts, key=cv2.contourArea,reverse=True)
                
                
                #To locate the bounding rectangle with the largest contour and 
                #use it to scale the resulting suit image to fit the size of the train suit image.
                if len(Fsuit_cnts) != 0:
                    x2,y2,w2,h2 = cv2.boundingRect(Fsuit_cnts[0])
                    Fsuit_roi = Fsuit[y2:y2+h2, x2:x2+w2]
                    Fsuit_sized = cv2.resize(Fsuit_roi, (Suit_W, Suit_H), 0, 0)
                    suit_img = Fsuit_sized
                
                
                
                rank_match_diff = 10000
                suit_match_diff = 10000
                rank_match_name = "Unknown"
                suit_match_name = "Unknown"
    
            
    
                
                #To obtain the difference between the resulting card rank image and each of the train rank images
                #and the result with the least difference is saved.
                for Trank in train_ranks:
        
                    diff_img = cv2.absdiff(rank_img, Trank.img)
                    rank_diff = int(np.sum(diff_img)/255)
                    #print(rank_diff)
                    
                    if rank_diff < rank_match_diff:
                        
                        best_rank_diff_img = diff_img
                        rank_match_diff = rank_diff
                        best_rank_name = Trank.name
                        #print(best_rank_name)
        
                #To find the least difference between the acquired card suit image and each of the train suit images
                #and save the result with the smallest difference.
                for Tsuit in train_suits:
                        
                        diff_img = cv2.absdiff(suit_img, Tsuit.img)
                        suit_diff = int(np.sum(diff_img)/255)
                        #print(suit_diff,Tsuit.name)
                        
                        if suit_diff < suit_match_diff:
                            best_suit_diff_img = diff_img
                            suit_match_diff = suit_diff
                            best_suit_name = Tsuit.name
        
            
            ##To identify the identity of the obatined card, we combine the best rank match with the best suit match.
            #If the best matches have an unusually significant difference value, 
            #the identity of the card remains unknown.
                if (rank_match_diff < Max_Rank_Diff):
                    rank_match_name = best_rank_name
            
                if (suit_match_diff < Max_Suit_Diff):
                    suit_match_name = best_suit_name
                    
            
                # Return the identiy of the card and the quality of the suit and rank match
                #print(rank_match_name, suit_match_name, rank_match_diff, suit_match_diff)
                #print(rank_match_name, suit_match_name)
            
    
            
      
    st.subheader('Image After Preprocessing')
    st.text("Operations : Greyscale Convertion , Gaussian Blurring & Thresholding")
    st.image(thresh_image,width = 400)
    
    
    st.subheader('After Perspective Transform')
    st.text("Obtained a 200x300 flattened image")
    st.image(warp,width = 400)
    
    st.subheader('Segmented Rank and Suit Image of Input image')
    col1, col2 = st.columns(2)
    with col1:
      st.image(rank_img)
    with col2:
      st.image(suit_img)
    
    st.write("##")
    st.write("##")
    
    
    
    
    
    
    
    
    
    
    
    st.write("##")
    st.subheader('Output Data')
    st.write('Given image is', rank_match_name , 'of ' ,suit_match_name )   
                    
                    
                    
                
                
                
                
        
