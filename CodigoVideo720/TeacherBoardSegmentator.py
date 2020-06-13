import cv2
import numpy as np
import xml.etree.ElementTree as et
import imutils
import sys
import os

class Teacher:
    def _init(self,sizeBlur,teacherSizeX,teacherSizeH,whiteLimit,minimumArea,imgBack):
        """ initialize variables to be used for the teacher
        Arguments:
              self.imgBack = imgBack -- video frame background image
              self.sizeBlur = sizeBlur -- blur mask´s size
              self.teacherSizeX = teacherSizeX -- maximum horizontal step that a teacher can make
              self.teacherSizeH = teacherSizeH-- maximum vertical jump that a teacher can make
              self.whiteLimit = whiteLimit -- white limit to be used for the threshold
              self.minimumArea = minimumArea -- minimum area that a contour need to have to be a teacher
              self.rectangleColor = (0, 0, 0) --  black color to draw a rectangle around a teacher
          """
        self.imgBack = imgBack
        self.sizeBlur = sizeBlur
        self.teacherSizeX = teacherSizeX
        self.teacherSizeH = teacherSizeH
        self.whiteLimit = whiteLimit
        self.minimumArea = minimumArea
        self.rectangleColor = (0, 0, 0)

    def setTeacherParam(self,imgBack):
        """ set teacher parameters and apply Gaussian filter for background image
        Arguments:
        self.imgBack = imgBack -- background image
        self.sizeBlur = 31 -- gaussianBlur size. Size can be modified if needed without rectifying other parts of the code
        self.teacherSizeX = 500 -- horizontal size of teacher´s screen (width)
        self.teacherSizeH = 1000 -- vertical size of teacher´s screen  (height)
        self.backImgBlur = cv2.GaussianBlur(self.imgBack, (self.sizeBlur,self.sizeBlur), 0) -- Background image after applying GaussianBlur function
        self.whiteLimit = 20 -- threshold limit
        self.minimumArea = 10000 -- minimum area to be used to control the contours
        self.rectangleColor = (0, 0, 0) -- black color of the rectangle drawn around the teacher once they are found. Used to develop the code
        """
        self.imgBack = imgBack
        self.sizeBlur = 31
        self.teacherSizeX = 500
        self.teacherSizeH = 1000
        self.backImgBlur = cv2.GaussianBlur(self.imgBack, (self.sizeBlur,self.sizeBlur), 0)
        self.whiteLimit = 20
        self.minimumArea = 10000
        self.rectangleColor = (0, 0, 0)
        return self.imgBack, self.sizeBlur, self.minimumArea, self.teacherSizeX, self.teacherSizeH, self.backImgBlur, self.whiteLimit, self.minimumArea, self.rectangleColor

    def getTeacher(self,frame,teacherOut):
        """ getTeacher out of the actual frame and write it in the final video (teacherOut) together with the previous analysed frames
        self.greyImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) -- transform frame to Gray scale colors
        self.greyBlurImg = cv2.GaussianBlur(self.greyImg, (self.sizeBlur, self.sizeBlur), 0)--  transform background image to gray scale colors
        minusImg = cv2.absdiff(self.greyBlurImg, self.backImgBlur)-- take away frame from background image to later compare the changes in search of the teacher
        binaryImg = cv2.threshold(minusImg, self.whiteLimit, 255, cv2.THRESH_BINARY)[1] --  binarize the image to later see where the contours are formed.
        contours = cv2.findContours(binaryImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)-- find contours from the previous image
        contours = imutils.grab_contours(contours)-- variable with all the contours found
        (x, y, w, h) = cv2.boundingRect(contour) -- get the position of the contour in the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.rectangleColor, 2)-- create a rectangle around the contour of the frame
        # cv2.imshow("video with teacher detected",frame) --  used to visualize the result while developing the code and check if it is correct
        cropped = frame[y:y + h, x:x + w] -- crop the teacher from the frame using the the coordinates of the previous boundingRect function
        croppedResize = cv2.resize(cropped, (self.teacherSizeX, self.teacherSizeH)) --  resize the teacher image to the desirable size
        f = croppedResize.copy() -- copy the image that belongs to the teacher
        teacherOut.write(f) --  write the teacher´s image
        """
        self.greyImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.greyBlurImg = cv2.GaussianBlur(self.greyImg, (self.sizeBlur, self.sizeBlur), 0)
        minusImg = cv2.absdiff(self.greyBlurImg, self.backImgBlur)
        binaryImg = cv2.threshold(minusImg, self.whiteLimit, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(binaryImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:

            if cv2.contourArea(contour) < self.minimumArea:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.rectangleColor, 2)
        cropped = frame[y:y + h, x:x + w]
        croppedResize = cv2.resize(cropped, (self.teacherSizeX, self.teacherSizeH))
        f = croppedResize.copy()
        teacherOut.write(f)
        return croppedResize, frame,teacherOut

    def copy(self):
        # function used to copy and return the teacher
        return self.copy()
####################################################
class Board:
    def _init_(self, A, B, C, D, width, height, W, H, Dim):
        """ initialize parameters of the board
        :param A: -- position up to the left corner of the board
        :param B: -- position up to the right corner of the board
        :param C: -- position down to the right corner of the board
        :param D: -- position down to the left corner of the board
        :param W: -- board´s  width
        :param H: -- board´s height
        :param Dim: -- array with width and height
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.width = width
        self.height = height
        self.W = W
        self.H = H
        self.Dim = Dim


    def setBoardParam(self):
        """ initialize parameters of the board using an xml file
               :param A: -- position up to the left corner of the board
               :param B: -- position up to the right corner of the board
               :param C: -- position down to the right corner of the board
               :param D: -- position down to the left corner of the board
               :param W: -- board´s  width
               :param H: -- board´s height
               :param Dim: -- array with width and height
               """

        xmlFile = et.parse("boardDoc/CAMERA.xml")
        root = xmlFile.getroot()
        ax = int(root[0].attrib['a_x'])
        ay = int(root[0].attrib['a_y'])
        A = [ax, ay]
        self.A = A
        bx = int(root[0].attrib['b_x'])
        by = int(root[0].attrib['b_y'])
        B = [bx, by]
        self.B = B
        cx = int(root[0].attrib['c_x'])
        cy = int(root[0].attrib['c_y'])
        C = [cx, cy]
        self.C = C
        dx = int(root[0].attrib['d_x'])
        dy = int(root[0].attrib['d_y'])
        D = [dx, dy]
        self.D = D
        width = int(root[1].attrib['width'])
        self.width = width
        height = int(root[1].attrib['height'])
        self.height = height
        W = max(cx - ax, dx - bx)
        self.W= W
        H = W * height / width
        self.H = H
        Dim = [W, H]
        self.Dim = Dim
        return A, B, C, D, width, height, W, H, Dim

    def getBoardValue(self):
        """
        return the parameters of the board
        """
        return self.A, self.B, self.C, self.D, self.Dim


    def Transform4Points(self,img):
        """
        The coordinates of the board do not belong to a rectangle because the camera is not completely perpendicular.
        Because of this, we need to make a transformation to rectify its position
        """
        #       A __________ B         (0,0)____________
        #      /          /               |            |
        #     /          /      ===>      |            |
        #   D/__________/C                |____________| (W,H)


        pts1 = np.float32([self.A, self.B, self.D, self.C])
        pts2 = np.float32([[0, 0], [self.Dim[0], 0], [0, self.Dim[1]], self.Dim])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (pts2[3, 0], pts2[3, 1]))
        return dst


    def findBoard(self, frame,boardOut):
        """
        this function calls the previous one resizing the return value according to our screen so it coincides with the height and width of the teacher
        """
        crop = self.Transform4Points(frame)
        cropResize = cv2.resize(crop, (1500, 700))
        f = cropResize.copy()
        boardOut.write(f)


        return cropResize, frame, boardOut


class Camera():

    def _init_(self,path):
        """ initialize the parameters of the Camera
        :param path: -- route the video takes.
        """
        self.path = path
        self.capture = cv2.videoCapture(0)

    def start(self,path):
        """

        :param path: route the video takes.
        """
        self.path = path
        self.capture = cv2.VideoCapture(self.path)


    def get_frame(self):
        """
        :return: -- return two variables: a boolean one to check if it has found the frame or not (s) and the actual frame
        """
        s, img = self.capture.read()
        if s:
            pass
        return s, img

    def release_camera(self):

        self.capture.release()

    def check_If_Opened(self):
        if  self.capture.isOpened() == False:
            print("Error opening video stream or file")
            exit()
        return True

########################################################################################3



def minhorizontalConcatResize(listOfImg, interpolation=cv2.INTER_CUBIC):
    """
    Join the image of the teacher with the image of the board once they have been detected
    in order to return one final screen of both of them together.
    Function used while developing the code to make the checking process more confortable
    :param listOfImg: -- array of teacher and board

    """
    hmin = min(im.shape[0] for im in listOfImg)
    imListResize = [cv2.resize(im, (int(im.shape[1] * hmin / im.shape[0]), hmin), interpolation=interpolation)
                      for im in listOfImg]
    return cv2.hconcat(imListResize)



def DrawPointsFrame(img,A,B,C,D):
    """
     receive an image and four points and return the image with the points connected and colored.
     Used while developing the code
    """
    lineColor = (255,255,255)
    width = 3
    cv2.line(img,(A[0],A[1]),(B[0],B[1]),lineColor,width)
    cv2.line(img, (A[0], A[1]), (B[0], B[1]), lineColor, width)
    cv2.line(img, (A[0], A[1]), (B[0], B[1]), lineColor, width)
    cv2.line(img, (A[0], A[1]), (B[0], B[1]), lineColor, width)
    return img


def main(videoPath,resultDir):

    board = Board()
    board.setBoardParam()
    firstBoard = 0
    path = r'C:\Users\Lara\Desktop\TFG_Prueba1\Fondo.png'
    imgBack = cv2.imread(path,0)
    teacher = Teacher()
    teacher.setTeacherParam(imgBack)
    capture = Camera()
    capture.start(videoPath)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    FPS = 24
    width = 500
    height = 1000
    widthBoard = 1500
    heightBoard = 700

    boardName = os.path.join(resultDir, "Board_CAMERA.avi")

    teacherName= os.path.join(resultDir,"teacher_CAMERA.avi")

    teacherOut = cv2.VideoWriter(teacherName,fourcc,float(FPS),(width,height))
    boardOut = cv2.VideoWriter(boardName,fourcc,float(FPS),(widthBoard,heightBoard))

    while capture.check_If_Opened():

        ret, frame = capture.get_frame()

        if ret:
            cropboard, frame, boardOut = board.findBoard(frame,boardOut)
            cropteacher,frame,teacherOut = teacher.getTeacher(frame,teacherOut)
            if firstBoard == 0:
                boardAux = cropboard.copy()
                firstBoard = 1


        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release_camera()
    teacherOut.release()
    boardOut.release()

if __name__ == '__main__':


   if len(sys.argv) != 3:
       raise ValueError('número de parámetros incorrecto')

   videoPath = sys.argv[1]

   resultDir = sys.argv[2]

   head, tail = os.path.split(videoPath)
   print("\nEjecutando proceso del video: ",tail," \n\nEspere unos instantes...")

   main(videoPath,resultDir)
   print("\n\nProceso terminado de forma satisfactoria.\n\nEl directorio donde se encuentran los resultados es el siguiente:\n\n",resultDir)

   cv2.destroyAllWindows()


