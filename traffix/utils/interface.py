from tkinter import Tk, Canvas
from PIL import Image, ImageTk, ImageShow




font = ("Sans Serif", 20)
x = 300



def photo_image(img, resize_to: tuple):
    image = Image.fromarray(img)
    image = image.resize(resize_to, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(image)




class DisplayCanvas:
    def __init__(self, root: Tk, width: int, height: int, pos: tuple, name: str) -> None:
        self._canvas = Canvas(root, width=width, height=height, background="gray")
        self._img_size = 328, 247
        self._image = self._canvas.create_image(x, 150)
        self._title = self._canvas.create_text(x, 300, text=name, font=font)
        self._current_flow = self._canvas.create_text(x, 330, font=font)
        self._estimated_flow = self._canvas.create_text(x, 360, font=font)

        self._canvas.grid(column=pos[0], row=pos[1])


    def update(self,
               img=None,
               current_flow: float | None = None,
               estimated_flow: float | None = None) -> None:
        if img is not None:
            self._tkimage = photo_image(img, self._img_size)
            self._canvas.itemconfig(self._image, image=self._tkimage)

        if current_flow is not None:
            self._canvas.itemconfig(self._current_flow, text=f"Current Flow: {current_flow}")

        if estimated_flow is not None:
            self._canvas.itemconfig(self._estimated_flow, text=f"Estimated Flow: {estimated_flow}")




class SumCanvas:
    def __init__(self, root: Tk, width: int, height: int, pos: tuple, name: str) -> None:
        self._canvas = Canvas(root, width=width, height=height, background="yellow")
        self._title = self._canvas.create_text(x, 20, text=name, font=font)
        self._current_flow = self._canvas.create_text(x, 50, font=font)
        self._estimated_flow = self._canvas.create_text(x, 80, font=font)
        self._green_light_time = self._canvas.create_text(x, 110, font=font)

        self._canvas.grid(column=pos[0], row=pos[1])


    def update(self,
               current_flow: float | None = None,
               estimated_flow: float | None = None,
               green_light_time: float | None = None) -> None:
        if current_flow is not None:
            self._canvas.itemconfig(self._current_flow, text=f"Current Flow: {current_flow}")

        if estimated_flow is not None:
            self._canvas.itemconfig(self._estimated_flow, text=f"Estimated Flow: {estimated_flow}")

        if green_light_time is not None:
            self._canvas.itemconfig(self._green_light_time, text=f"Green Light Time: {green_light_time}")




class Interface:
    def __init__(self) -> None:
        self._root = Tk()
        self._root.geometry("1200x1000")
        self._root.resizable(False, False)
        self._root.title("Traffix")
        self._display_canvases = [DisplayCanvas(self._root, 600, 400, (int(i / 2), i % 2), name) for i, name in enumerate(["north", "south", "east", "west"])]
        self._sum_canvases = [SumCanvas(self._root, 600, 200, (i, 2), name) for i, name in enumerate(["north-south", "east-west"])]

    
    def get_display_canvas(self, idx: int) -> Canvas:
        return self._display_canvases[idx]


    def get_sum_canvas(self, idx: int) -> Canvas:
        return self._sum_canvases[idx]


    def run_main_loop(self) -> None:
        self._root.mainloop()
