import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from tensorflow import expand_dims
import numpy as np
from keras.models import load_model


# Load pre-trained model
model = load_model('Electric_Component_v3.0.h5')

#Tạo mảng để chứa label
label = np.array(['IC', 'SMD Capacitor', 'SMD Resistor', 'SMD Transistor',
                  'ThroughHole Capacitor', 'ThroughHole Resistor', 
                  'ThroughHole Transistor'])

#Tạo mảng để chứa Thông tin linh kiện
detail = np.array(['IC (Integrated Circuit) là một thành phần điện tử được sản xuất trên một mảng nhỏ của vật liệu bán dẫn, chứa hàng triệu hoặc thậm chí hàng tỷ thành phần điện tử như transistor, điện trở và điốt.',
                    'SMD Capacitor (Surface Mount Device Capacitor) là loại tụ điện được thiết kế để gắn trực tiếp trên bề mặt PCB (Printed Circuit Board).',
                    'SMD Resistor (Surface Mount Device Resistor) là loại điện trở được thiết kế để gắn trực tiếp trên bề mặt PCB (Printed Circuit Board).',
                    'SMD Transistor (Surface Mount Device Transistor) là loại transistor được thiết kế để gắn trực tiếp trên bề mặt PCB (Printed Circuit Board).',
                    'Through Hole Capacitor (tạm dịch là Tụ điện lỗ thông) là loại tụ điện được thiết kế để lắp đặt trên bề mặt lỗ thông của PCB (Printed Circuit Board).',
                    'Through Hole Resistor (tạm dịch là Điện trở lỗ thông) là loại điện trở được thiết kế để lắp đặt trên bề mặt lỗ thông của PCB (Printed Circuit Board).', 
                    'Through Hole Transistor (tạm dịch là Transistor lỗ thông) là loại transistor được thiết kế để lắp đặt trên bề mặt lỗ thông của PCB (Printed Circuit Board).'])

class MyWindow:
    def __init__(self, master):
        self.master = master
        master.title("Giao diện")

        # Các thành phần giao diện
        self.label = tk.Label(master)
        self.text_detail = tk.Text(master, height=5, width=50)
        self.text_detail.insert(tk.END, "Thông tin linh kiện sẽ hiển thị ở đây")
        self.text_detail.config(state="disabled")
        self.text_detail.config(bd=0, highlightbackground="white", bg="white")
        self.button_load = tk.Button(master, text="Tải ảnh",bg="red", fg="white", command=self.load_image)
        self.button_start = tk.Button(master, text="Bật camera",bg="blue", fg="white", command=self.start_camera)
        self.button_stop = tk.Button(master, text="Tắt camera", command=self.stop_camera)
        self.camera_running = False

        #Thong tin ung dung
        self.text_info = tk.Text(master, height=1, width=120)
        self.text_info.insert(tk.END, "NHẬN DIỆN MỘT SỐ LOẠI LINH KIỆN ĐIỆN TỬ SỬ DỤNG MẠNG CNN")
        self.text_info.config(state="disabled", font=("Arial", 16), fg="red")

        # Layout
        self.label.place(x=100, y=100)
        self.text_detail.place(x=100, y=450)
        self.text_info.place(x=30,y=30)
        self.button_load.place(x=550, y=100, width=100, height=50)
        self.button_start.place(x=550, y=160, width=100, height=50)
        self.button_stop.place(x=550, y=220, width=100, height=50)

        #Hien anh ban dau
        image_init = Image.open("Info.jpg").resize((300, 300))
        photo = ImageTk.PhotoImage(image_init)
        self.label.config(image=photo)
        self.label.image = photo

        #Nút thoát ứng dụng
        # Thiết lập sự kiện WM_DELETE_WINDOW khi người dùng nhấn nút Close
        root.protocol("WM_DELETE_WINDOW", root.quit)

        # Tạo một button Exit và thiết lập sự kiện command
        self.button_exit = tk.Button(root,bg="cyan", text="EXIT", command=root.destroy)
        self.button_exit.place(x= 550, y= 350, width=100, height=50)

    def load_image(self):
        # Mở hộp thoại để chọn ảnh
        file_name = tkinter.filedialog.askopenfilename(filetypes=[('Image Files', ('*.jpg', '*.jpeg', '*.png', '*.bmp'))])

        if file_name:
            # Mở ảnh 
            image_original = Image.open(file_name)

            # Chuyển đổi ảnh về dạng numpy
            image = np.array(image_original)
            image = cv2.resize(image, (160, 160))
            image = image / 255.0
            image = expand_dims(image, axis=0)

            # Thực hiện dự đoán
            max = np.argmax(model.predict(image), axis=1)

            # Hiển thị chi tiết về linh kiện
            self.text_detail.config(state="normal")
            self.text_detail.delete("1.0", tk.END)
            self.text_detail.insert(tk.END, str(detail[max]))
            self.text_detail.config(state="disabled")

            # Resize ảnh để dễ hiển thị
            image_resized = image_original.resize((300, 300))

            #Chọn ảnh cần vẽ cho lệnh draw từ PIL
            draw = ImageDraw.Draw(image_resized)

            # Thiết lập font chữ và kích thước
            font = ImageFont.truetype("arial.ttf", 20)

            # Viết chữ với font và kích thước đã thiết lập
            draw.text((0, 0), str(label[max]), fill=(255, 0, 0), font=font)

            # Hiển thị ảnh lên label
            photo = ImageTk.PhotoImage(image_resized)
            self.label.config(image=photo)
            self.label.image = photo

    def start_camera(self):
        if not self.camera_running:
            # Mở camera
            self.cap = cv2.VideoCapture(1)
            self.camera_running = True
            self.update_frame()

    def stop_camera(self):
        if self.camera_running:
            # Dừng camera
            self.cap.release()
            self.camera_running = False

    def update_frame(self):
        if self.camera_running:
            ret, frame = self.cap.read()
            # Xử lí ảnh đưa về dạng numpy
            image = cv2.resize(frame, (160, 160))
            image = image / 255.0
            image = expand_dims(image, axis=0)

            # Thực hiện dự đoán
            max = np.argmax(model.predict(image), axis = 1)

            # Hiển thị dự đoán trực tiếp lên hình ảnh từ Camera
            cv2.putText(frame, str(label[max]), (50, 50 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Hiển thị chi tiết về linh kiện
            self.text_detail.config(state="normal")
            self.text_detail.delete("1.0", tk.END)
            self.text_detail.insert(tk.END, str(detail[max]))
            self.text_detail.config(state="disabled")

            if ret:
                # Hiển thị hình ảnh lên label
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image = image.resize((300, 300), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                self.label.config(image=photo)
                self.label.image = photo
                self.master.after(5, self.update_frame)

if __name__ == '__main__':
    root = tk.Tk()
    
    # Thiết lập kích thước của cửa sổ
    root.geometry("800x600")

    # Mở ảnh và chuyển đổi thành định dạng Tkinter
    image = Image.open("nen.jpg")
    photo = ImageTk.PhotoImage(image)

    # Tạo một Canvas widget và vẽ ảnh lên nền của canvas
    canvas = tk.Canvas(root, width=800, height=500)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    window = MyWindow(root)
    root.mainloop()