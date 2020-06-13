import cv2
import numpy as np
import xml.etree.ElementTree as et
import imutils
import os
import sys
class Teacher:

    def setTeacherParam(self,imgBack):
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
        self.sizeBlur = 31
        self.teacherSizeX = 500
        self.teacherSizeH = 1000
        self.backImgBlur = cv2.GaussianBlur(imgBack, (self.sizeBlur,self.sizeBlur), 0)
        self.backImgBlur = cv2.resize(self.backImgBlur, (3840, 2160))
        self.whiteLimit = 80
        self.minimumArea = 10000
        self.rectangleColor = (0, 0, 0)
        return self.imgBack, self.sizeBlur, self.minimumArea, self.teacherSizeX, self.teacherSizeH, self.backImgBlur, self.whiteLimit, self.minimumArea, self.rectangleColor

    def getTeacher(self,frame, teacher_previous_x, teacher_previous_y,teacherOut):
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
            if teacher_previous_x and teacher_previous_y:
               if abs(y - teacher_previous_y)<200 and abs(x - teacher_previous_x)<200:
                   break

            cv2.rectangle(frame, (x, y), (x + w, y + h), self.rectangleColor, 2)

        cropped = frame[y:y + h, x:x + w]
        croppedResize = cv2.resize(cropped, (self.teacherSizeX, self.teacherSizeH))
        f=croppedResize.copy()
        teacherOut.write(f)
        return croppedResize, frame, x, y, teacherOut

    def copy(self):
        return self.copy()
class Board:
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
        xmlFile = et.parse("boardDoc/CAMERA2.xml")
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
        return cropResize,frame, boardOut


class Camera():
    def start(self,path):
        """ initialize the parameters of the Camera
              :param path: -- route the video takes.
              """
        self.path = path
        self.capture = cv2.VideoCapture(self.path)
        self.s = False
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

def teacherInOutBoard(teacher,board):
    """

    :param teacher: -- recieve a teacher frame
    :param board: recieve a board frame
    check if the teacher is in the board
    used to develop the code
    """
    img = teacher.copy()
    img = cv2.resize(img, (500,1000))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = img.copy()
    template = board.copy()
    template = cv2.resize(template, (1500, 1000))
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

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
def ImgFilter(board):

    median = cv2.medianBlur(board, 3)
    return median


    return median

def movimientoProfesor(auxcropped, cropped, newx, newy, teacherStepX, falseTeacherY):

        if ((teacherStepX + 200) >= newx and (teacherStepX - 200) <= newx):

            teacherStepX = newx
            falseTeacherY = newy
            auxcropped = cropped.copy()
        else:
            cropped = auxcropped.copy()

        return auxcropped, cropped, teacherStepX, falseTeacherY


auxFrame = 0


def parecidoEnImagen(auxcropped, cropped, x, y, w, h):
    template = cv2.cvtColor(auxcropped, cv2.COLOR_BGR2GRAY)
    newtemplate = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    template = cv2.resize(template, (1500, 1000))
    newtemplate = cv2.resize(newtemplate, (1500, 1000))
    res = cv2.matchTemplate(newtemplate, template, cv2.TM_CCOEFF_NORMED)
    if res < 0.70:
        cropped = auxcropped.copy()
        auxcropped = cropped.copy()
    else:
        auxcropped = cropped.copy()
    return cropped, auxcropped



def main(videoPath,resultDir):

    board = Board()
    board.setBoardParam()
    firstBoard = 0
    path = r'C:\Users\Lara\Desktop\TFG_Prueba2\fondoDos.png'
    imgBack = cv2.imread(path, 0)
    teacher = Teacher()
    teacher.setTeacherParam(imgBack)
    capture = Camera()
    capture.start(videoPath)
    teacher_previous_x = None
    teacher_previous_y = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    FPS = 24
    width = 500
    height = 1000
    widthBoard = 1500
    heightBoard = 700
    x = None
    y = None
    boardName = os.path.join(resultDir, "Board_PizarraMitadLuzDeTarde.avi")

    teacherName = os.path.join(resultDir, "teacher_PizarraMitadLuzDeTarde.avi")


    teacherOut = cv2.VideoWriter(teacherName,fourcc,float(FPS),(width,height))
    boardOut = cv2.VideoWriter(boardName,fourcc,float(FPS),(widthBoard,heightBoard))

    while capture.check_If_Opened():
        ret, frame = capture.get_frame()
        if ret:
            cropboard, frame, boardOut = board.findBoard(frame,boardOut)
            cropteacher,frame,x,y, teacherOut = teacher.getTeacher(frame,teacher_previous_x,teacher_previous_y,teacherOut)
            if firstBoard == 0:
                boardAux = cropboard.copy()
                firstBoard = 1

            teacher_previous_x = x
            teacher_previous_y = y
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
   print("video Path: ",videoPath)
   videoPath = os.path.join(videoPath+"."+"MP4")
   print("videoPath mp4:",videoPath)
   print("\nEjecutando proceso del video: ",tail," \n\nEspere unos instantes...")
   print(tail)
   main(videoPath,resultDir)
   print("\n\nProceso terminado de forma satisfactoria.\n\nEl directorio donde se encuentran los resultados es el siguiente:\n\n",resultDir)

   cv2.destroyAllWindows()