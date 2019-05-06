import numpy as np
import cv2

FINAL_LINE_COLOR = (0, 0, 0)
WORKING_LINE_COLOR = (255, 255, 255)

class Draw(object):
    def __init__(self, window_name, clean_img=None, img_size=None):
        self.window_name = window_name # Name for our window
        self.img_size = img_size
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.clean_img = clean_img

    def set_clean_img(self, clean_img):
        self.clean_img = clean_img.copy()

    def set_img_size(self, img_size):
        self.img_size = img_size

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.curr_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True


    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, np.zeros(self.img_size, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        if self.clean_img is None:
            canvas = np.ones(self.img_size, np.uint8)
        else:
            canvas = self.clean_img
        while(not self.done):
            self.curr_points = []
            while(True):
                if (len(self.curr_points) > 0):
                    cv2.polylines(canvas, np.array([self.curr_points]), False, FINAL_LINE_COLOR, 1)
                    # cv2.line(canvas, self.curr_points[-1], self.current, WORKING_LINE_COLOR)
                cv2.imshow(self.window_name, canvas)
                if cv2.waitKey(100) == 27: # ESC hit
                    break
            self.points.append(self.curr_points)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        canvas = np.ones(self.img_size, np.float64)
        for points in self.points:
            if (len(points) > 0):
                cv2.fillPoly(canvas, np.array([points]), FINAL_LINE_COLOR)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas

if __name__ == "__main__":
    pd = Draw("Polygon")
    image = pd.run()
    cv2.imwrite("polygon.png", image)
    print("Polygon = %s" % pd.points)
