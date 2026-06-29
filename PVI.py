import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, simpledialog
import napari
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure
import pandas as pd
import os
import zipfile
import gc


vol_original_global = None
vol_gauss_global = None
mu_final_global = None 
mu_original_global = None  
vi_global = None
video_data_global = None

ruta_archivo_global = ""


def leer_y_procesar_oct(ruta_archivo_oct):
    global vol_original_global
    global vol_gauss_global
    global mu_final_global
    global mu_original_global  
    global vi_global
    global video_data_global

    try:
        print(f"Abriendo {os.path.basename(ruta_archivo_oct)} ")

        with zipfile.ZipFile(ruta_archivo_oct, 'r') as z:
            with z.open('data/Intensity.data') as f:
                vol_original = np.frombuffer(
                    f.read(),
                    dtype=np.float32
                ).reshape((512, 1024, 512)).copy()

            with z.open('data/VideoImage.data') as f:
                video_data = np.frombuffer(
                    f.read(),
                    dtype=np.uint8
                ).reshape((640, 480, 4)).copy()

        print("Aplicando filtro gaussiano...")
        vol_gauss = gaussian_filter(
            vol_original,
            sigma=1
        ).astype(np.float32)

        print("Aplicando iluminación...")
        v_min_g, v_max_g = vol_gauss.min(), vol_gauss.max()
        vol_norm = (vol_gauss - v_min_g) / (v_max_g - v_min_g)
        vi = exposure.equalize_adapthist(
            vol_norm,
            kernel_size=128,
            clip_limit=0.1
        ).astype(np.float32)

        # ---- CÁLCULO DE COEFICIENTE DE ATENUACIÓN ----
        dz = 1.71878 / 512
        axis_z = 2

       
        print("Aplicando coeficiente de atenuación (Filtro Gauss)...")
        vol_limpio_gauss = np.maximum(vol_gauss, 1e-8)
        cum_int_gauss = np.cumsum(
            vol_limpio_gauss[:, :, ::-1],
            axis=axis_z
        )[:, :, ::-1]
        alpha_gauss = np.median(cum_int_gauss) * 1.3
        
        mu_vermeer_gauss = vol_limpio_gauss / (
            2.0 * dz * cum_int_gauss + alpha_gauss + 1e-10
        )
        v_max_mu_gauss = np.percentile(mu_vermeer_gauss, 98)
        mu_final = np.clip(
            mu_vermeer_gauss,
            0,
            v_max_mu_gauss
        ).astype(np.float32)

        
        print("Aplicando coeficiente de atenuación (Volumen Original)...")
        vol_limpio_orig = np.maximum(vol_original, 1e-8)
        cum_int_orig = np.cumsum(
            vol_limpio_orig[:, :, ::-1],
            axis=axis_z
        )[:, :, ::-1]
        alpha_orig = np.median(cum_int_orig) * 1.3
        
        mu_vermeer_orig = vol_limpio_orig / (
            2.0 * dz * cum_int_orig + alpha_orig + 1e-10
        )
        v_max_mu_orig = np.percentile(mu_vermeer_orig, 98)
        mu_original = np.clip(
            mu_vermeer_orig,
            0,
            v_max_mu_orig
        ).astype(np.float32)
        # -----------------------------------------------

       
        vol_original_global = vol_original
        vol_gauss_global = vol_gauss
        mu_final_global = mu_final
        mu_original_global = mu_original 
        vi_global = vi
        video_data_global = video_data

       
        viewer = napari.Viewer()


        layer1 = viewer.add_image(
            video_data,
            name="zona de captura de datos"
        )
        
        layer2 = viewer.add_image(
            vol_original,
            name='volumen original',
            colormap='viridis',
            gamma=2
        )
        
        layer3 = viewer.add_image(
            mu_original,
            name='coeficiente de atenuacion',
            colormap='viridis',
            gamma=2
        )
        layer4 = viewer.add_image(
            vol_gauss,
            name='filtro gauss',
            colormap='viridis',
            gamma=2,
            blending='translucent',
            rendering='iso',
            iso_threshold=45
        )
        layer5 = viewer.add_image(
            vi,
            name='filtro gauss e iluminacion',
            colormap='viridis',
            gamma=2,
            rendering='translucent'
        )
        layer6 = viewer.add_image(
            mu_final,
            name='filtro gauss y coeficiente de atenuacion',
            colormap='viridis',
            gamma=2
        )
        
        #este layer muestra el mapa peroa colores donde se pueden apreciar algunos detalles extra
        #pero en algunos oct no se bien devido a la tecnica que se utiliza para la toma de la muestra 
        #layer7 = viewer.add_labels(
        #    vol_gauss.astype(np.int16),
        #    name='mapa de color',
        #    opacity=1.0
        #)
        

        layer1.name_overlay.visible = True
        layer2.name_overlay.visible = True
        layer3.name_overlay.visible = True
        layer4.name_overlay.visible = True
        layer5.name_overlay.visible = True  
        layer6.name_overlay.visible = True
        #layer7.name_overlay.visible = True

        viewer.dims.order = (1, 0)

        viewer.grid.enabled = True
        viewer.grid.shape = (2, 3) 

        print("Visualizador listo.")
        napari.run()

        del mu_vermeer_gauss, mu_vermeer_orig
        gc.collect()

    except KeyError:
        messagebox.showerror(
            "Error",
            "La estructura interna del .OCT no coincide."
        )
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Ocurrió un error:\n{str(e)}"
        )


def normalizar_imagen(img):
    img = img.astype(np.float32)
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img


def exportar_resultados():
    global vol_original_global
    global vol_gauss_global
    global mu_final_global
    global mu_original_global 
    global vi_global

    if vol_original_global is None:
        messagebox.showwarning(
            "Aviso",
            "Primero debes visualizar un archivo."
        )
        return

    ventana_vista = tk.Toplevel()
    ventana_vista.title("Seleccionar Vista")
    ventana_vista.geometry("300x250")

    vista_seleccionada = tk.StringVar()
    vista_seleccionada.set("Z")

    tk.Label(
        ventana_vista,
        text="Seleccione la vista:",
        font=("Arial", 14, "bold")
    ).pack(pady=20)

    tk.Radiobutton(
        ventana_vista,
        text="Vista Z (Axial)",
        variable=vista_seleccionada,
        value="Z"
    ).pack(anchor="w", padx=30)

    tk.Radiobutton(
        ventana_vista,
        text="Vista Y (Coronal)",
        variable=vista_seleccionada,
        value="Y"
    ).pack(anchor="w", padx=30)

    tk.Radiobutton(
        ventana_vista,
        text="Vista X (Sagital)",
        variable=vista_seleccionada,
        value="X"
    ).pack(anchor="w", padx=30)

    def continuar_exportacion():
        vista = vista_seleccionada.get()
        ventana_vista.destroy()

        if vista == "Z":
            total_cortes = vol_original_global.shape[0]
        elif vista == "Y":
            total_cortes = vol_original_global.shape[1]
        else:
            total_cortes = vol_original_global.shape[2]

        corte_usuario = simpledialog.askinteger(
            "Seleccionar Corte",
            f"Ingrese número de corte para vista {vista}\n"
            f"(0 - {total_cortes - 1})"
        )

        if corte_usuario is None:
            return

        if corte_usuario < 0 or corte_usuario >= total_cortes:
            messagebox.showerror(
                "Error",
                "Número de corte inválido."
            )
            return

        carpeta = filedialog.askdirectory(
            title="Selecciona carpeta de exportación"
        )
        if not carpeta:
            return

        try:
            if vista == "Z":
                corte_original = vol_original_global[corte_usuario, :, :]
                corte_gauss = vol_gauss_global[corte_usuario, :, :]
                corte_mu = mu_final_global[corte_usuario, :, :]
                corte_mu_orig = mu_original_global[corte_usuario, :, :]  
                corte_vi = vi_global[corte_usuario, :, :]
            elif vista == "Y":
                corte_original = vol_original_global[:, corte_usuario, :]
                corte_gauss = vol_gauss_global[:, corte_usuario, :]
                corte_mu = mu_final_global[:, corte_usuario, :]
                corte_mu_orig = mu_original_global[:, corte_usuario, :]  
                corte_vi = vi_global[:, corte_usuario, :]
            else:
                corte_original = vol_original_global[:, :, corte_usuario]
                corte_gauss = vol_gauss_global[:, :, corte_usuario]
                corte_mu = mu_final_global[:, :, corte_usuario]
                corte_mu_orig = mu_original_global[:, :, corte_usuario]  
                corte_vi = vi_global[:, :, corte_usuario]

            corte_original = normalizar_imagen(corte_original)
            corte_gauss = normalizar_imagen(corte_gauss)
            corte_mu = normalizar_imagen(corte_mu)
            corte_mu_orig = normalizar_imagen(corte_mu_orig) 
            corte_vi = normalizar_imagen(corte_vi)

            # Guardar imágenes en disco
            Image.fromarray(corte_original).save(
                os.path.join(carpeta, f"{vista}_CORTE_{corte_usuario}_ORIGINAL.png")
            )
            Image.fromarray(corte_gauss).save(
                os.path.join(carpeta, f"{vista}_CORTE_{corte_usuario}_GAUSS.png")
            )
            Image.fromarray(corte_mu).save(
                os.path.join(carpeta, f"{vista}_CORTE_{corte_usuario}_ATENUACION.png")
            )
            Image.fromarray(corte_mu_orig).save(
                os.path.join(carpeta, f"{vista}_CORTE_{corte_usuario}_ATENUACION_ORIGINAL.png")
            )
            Image.fromarray(corte_vi).save(
                os.path.join(carpeta, f"{vista}_CORTE_{corte_usuario}_ILUMINACION.png")
            )

            messagebox.showinfo(
                "Exportación Exitosa",
                f"Vista: {vista}\n"
                f"Corte: {corte_usuario}\n\n"
                f"Exportación completada de manera exitosa."
            )

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"No se pudo exportar:\n{str(e)}"
            )

    tk.Button(
        ventana_vista,
        text="Continuar",
        bg="lime green",
        command=continuar_exportacion
    ).pack(pady=20)


def seleccionar_archivo():
    global ruta_archivo_global
    archivo = filedialog.askopenfilename(
        title="Selecciona archivo .OCT",
        filetypes=[("Archivos OCT", "*.OCT")]
    )
    if archivo:
        ruta_archivo_global = archivo
        texto_contenedor_boton.config(
            text=f"LISTO: {os.path.basename(archivo)}",
            fg="green"
        )


def visualizar():
    if ruta_archivo_global:
        leer_y_procesar_oct(ruta_archivo_global)
    else:
        messagebox.showwarning(
            "Aviso",
            "Por favor carga primero un archivo .OCT"
        )



def mostrar_instrucciones():
    ventana_instrucciones = tk.Toplevel()
    ventana_instrucciones.title("Instrucciones de Uso")
    ventana_instrucciones.geometry("680x420")
    ventana_instrucciones.config(background="sky blue")
    
   
    frame_interno = tk.Frame(ventana_instrucciones, relief="groove", borderwidth=3, bg="white")
    frame_interno.pack(padx=20, pady=20, fill="both", expand=True)
    
    tk.Label(
        frame_interno,
        text="Guía de Operación del Sistema",
        font=("TimesNewRoman", 16, "bold"),
        bg="white",
        fg="black"
    ).pack(pady=15)

    
    texto_instrucciones = (
        "1.- El programa solo admite y procesa archivos con terminación .OCT\n\n"
        "2.- Para poder cargar un archivo el usuario deberá darle al botón naranja que dice "
        "\"seleccionar el archivo .OCT\"\n\n"
        "3.- Una vez cargado deberá darle al botón verde que dice \"Visualizar resultados\"\n\n"
        "4.- Una vez dado al botón este abrirá otra interfaz donde se podrá tanto visualizar "
        "como manipular los resultados\n\n"
        "5.- Para poder exportar los resultados primero deberá subir el archivo, después ejecutar el análisis. "
        "Después de que se abra la interfaz del visualizador cerrarla, y ahora si darle al botón de "
        "\"exportar resultados\" donde el usuario podrá escoger de que vista el número de corte que quiere exportar.\n\n"
        "6.- Para poder hacer otro análisis o en caso de haberse confundido y quiera cambiar el archivo, puede darle "
        "al botón de \"borrar archivo\" para quitar el archivo actual cargado y seleccionar uno nuevo."
    )
    
    lbl_cuerpo = tk.Label(
        frame_interno,
        text=texto_instrucciones,
        font=("Arial", 11),
        justify="left",
        wraplength=600,
        bg="white",
        fg="black"
    )
    lbl_cuerpo.pack(padx=20, pady=5, anchor="w")
    
    tk.Button(
        ventana_instrucciones,
        text="Entendido",
        bg="coral",
        font=("Arial", 10, "bold"),
        command=ventana_instrucciones.destroy,
        cursor="hand2"
    ).pack(pady=10)


# Interfaz Gráfica Principal (Tkinter)
ventana_principal = tk.Tk()
ventana_principal.geometry("800x650")
ventana_principal.title("Extractor y Mejora OCT")
ventana_principal.config(background="sky blue")

try:
    logo1f = ImageTk.PhotoImage(Image.open("logo_UV.png").resize((100, 100)))
    logo2f = ImageTk.PhotoImage(Image.open("logo_laboratorio.png").resize((250, 100)))
    logo3f = ImageTk.PhotoImage(Image.open("logo_ITSX.png").resize((100, 100)))

    label_logos = tk.Label(ventana_principal, background="sky blue")
    label_logos.pack(anchor="n", fill="x", pady=10)

    tk.Label(label_logos, image=logo1f, bg="sky blue").pack(side="left", padx=20)
    tk.Label(label_logos, image=logo2f, bg="sky blue").pack(side="left", expand=True)
    tk.Label(label_logos, image=logo3f, bg="sky blue").pack(side="right", padx=20)
except:
    print("Logos no encontrados.")

contenedor_boton = tk.Frame(ventana_principal, relief="sunken", borderwidth=5)
contenedor_boton.pack(pady=40, padx=50, fill="x")

texto_contenedor_boton = tk.Label(
    contenedor_boton,
    text="CARGA DE ARCHIVO .OCT",
    font=("TimesNewRoman", 18, "bold")
)
texto_contenedor_boton.pack(pady=20)

tk.Button(
    contenedor_boton,
    text="Seleccionar archivo .OCT",
    bg="coral",
    command=seleccionar_archivo,
    cursor="hand2"
).pack(pady=10)



texto_contenedor = tk.Label(
    ventana_principal,
    text="\nPROGRAMA CREADO POR:\nImanol De la Garza Sanchez\n\nASESORADO POR:\nHector Hugo Cerecedo Núñez y Iluicatl Tonatiuh Villarreal Meza\n\nprograma orginal creado para el laboratorio de optica aplicada apoyado del visualizador de Napiri",
    bg= "sky blue",
    font=("TimesNewRoman",11, "bold italic")
).pack(pady=10, side="bottom")

contenedor_botones = tk.Frame(ventana_principal, relief="sunken", borderwidth=5)
contenedor_botones.pack(side="bottom", fill="x", padx=50, pady=20)

tk.Button(
    contenedor_botones,
    text="Visualizar resultados",
    bg="lime green",
    command=visualizar,
    cursor="hand2"
).pack(side="left", expand=True, padx=5, pady=10)

tk.Button(
    contenedor_botones,
    text="Exportar resultados",
    bg="gold",
    command=exportar_resultados,
    cursor="hand2"
).pack(side="left", expand=True, padx=5, pady=10)

tk.Button(
    contenedor_botones,
    text="Borrar archivo",
    bg="red",
    cursor="hand2",
    command=lambda: texto_contenedor_boton.config(
        text="CARGA DE ARCHIVO .OCT",
        fg="black"
    )
).pack(side="left", expand=True, padx=5, pady=10)


tk.Button(
    contenedor_botones,
    text="Instrucciones",
    bg="medium purple",
    cursor="hand2",
    command=mostrar_instrucciones
).pack(side="left", expand=True, padx=5, pady=10)

ventana_principal.mainloop()