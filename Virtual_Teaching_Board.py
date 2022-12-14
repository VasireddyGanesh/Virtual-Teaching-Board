import mediapipe as mp
import pyautogui
import cv2
import numpy as np
import time
import img2pdf
import pickle
import os

#contants
ml = 150
max_x, max_y = 350+ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0,0
screenshot_count=0
classno=0

#get tools function
def getTool(x):
	if x < 50 + ml:
		return "line"

	elif x<100 + ml:
		return "rectangle"

	elif x < 150 + ml:
		return"draw"

	elif x<200 + ml:
		return "circle"

	elif x<250 + ml:return "erase"
    
	elif x<300 +ml: return "Clear Screen"
    
	else:return "screenshot"

def index_raised(yi, y9):
	if (y9 - yi) > 40:
		return True

	return False

def getVariable():
		with open('file.pkl', 'rb') as file:
			classno = pickle.load(file)
		return classno

hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# drawing tools
tools = cv2.imread("final.jpg")
tools = tools.astype('uint8')

mask = np.ones((480, 640))*255
mask = mask.astype('uint8')

cap = cv2.VideoCapture(0)
while True:
	_, frm = cap.read()
	frm = cv2.flip(frm, 1)

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = hand_landmark.process(rgb)
	window_name="Virtual Teaching Board"
	# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
	# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
			x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)

			if x < max_x + 50 and y < max_y and x > ml:
				if time_init:
					ctime = time.time()
					time_init = False
				ptime = time.time()

				cv2.circle(frm, (x, y), rad, (0,255,255), 2)
				rad -= 1

				if (ptime - ctime) > 0.8:
					curr_tool = getTool(x)
					print("your current tool set to : ", curr_tool)
					time_init = True
					rad = 40
					if curr_tool == "screenshot" :
						image = pyautogui.screenshot()
						image = cv2.cvtColor(np.array(image),
											cv2.COLOR_RGB2BGR)

						if screenshot_count==0:
							classno = getVariable()
							parentDir=r"D:\Problem Solving\Node Projects\Virtual Teaching Board"
							dir="class "+str(classno)
							path=os.path.join(parentDir,dir)
							os.mkdir(path)
							os.chdir(path)
						cv2.imwrite("image"+str(screenshot_count)+".png", image)
						screenshot_count+=1

			else:
				time_init = True
				rad = 40

			if curr_tool == "draw":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
					prevx, prevy = x, y

				else:
					prevx = x
					prevy = y

			elif curr_tool == "line":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.line(frm, (xii, yii), (x, y), (50,152,255), thick)

				else:
					if var_inits:
						cv2.line(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "rectangle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(frm, (xii, yii), (x, y), (0,255,255), thick)

				else:
					if var_inits:
						cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "circle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.circle(frm, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

				else:
					if var_inits:
						cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
						var_inits = False

			elif curr_tool == "erase":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.circle(frm, (x, y), 50, (0,0,0,0.1), -1)
					cv2.circle(mask, (x, y), 50, 255, -1)
			elif curr_tool == "Clear Screen":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.rectangle(frm, (0, 0),(640,480), (0,0,0,0.1), -1)
					cv2.rectangle(mask, (0, 0),(640,480),255, -1)
                
	op = cv2.bitwise_and(frm, frm, mask=mask)
	frm[:, :, 1] = op[:, :, 1]
	frm[:, :, 2] = op[:, :, 2]
	frm[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frm[:max_y, ml:max_x], 0.3, 0)
	cv2.putText(frm, curr_tool, (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
	cv2.imshow(window_name, frm)
	cv2.imshow("temp",mask)
	if cv2.waitKey(1) == 27:
		if screenshot_count>0:
			parentDir=r"D:\Problem Solving\Node Projects\Virtual Teaching Board\class "+str(classno)
			dir="file.pdf"
			pdf_path=os.path.join(parentDir,dir)
			file = open(pdf_path, "wb")
			img =[]
			for i in range(screenshot_count):
				img_path = r"D:\Problem Solving\Node Projects\Virtual Teaching Board\class "+str(classno)+"\\image"+str(i)+".png"
				img.append(img_path)
			pdf_bytes = img2pdf.convert(img)
			file.write(pdf_bytes)
			file.close()
			print("Successfully made pdf file")
			classno+=1
			os.chdir('..')
			with open('file.pkl', 'wb') as file:
				pickle.dump(classno, file)		
		cv2.destroyAllWindows()
		cap.release()
		break
 