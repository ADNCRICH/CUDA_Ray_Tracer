# CUDA Ray Tracing

Result of each chapter in the ebook `"Ray Tracing in One Weekend"` by `Peter Shirley`.

## Simple Gradient Image

![Simple Gradient Image](./ch_01_first_image/first_image_cuda.jpg)

## Gradient Background

![Gradient Background](./ch_03_ray/background.jpg)

## Simple Sphere

![Simple Sphere](./ch_04_sphere/sphere.jpg)

## Spheres with Normal Map

![Spheres with Normal Map](./ch_05_normal_map/sphere_world.jpg)

## Anti-Aliasing

<!-- **Without Anti-Aliasing** | **With Anti-Aliasing**
--- | --- -->
<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; text-align: center;">
    <strong>Without Anti-Aliasing</strong><br>
    <img src="./ch_05_normal_map/sphere_world_cropped.jpg" alt="Without Anti-Aliasing" style="width: 95%; height: auto;">
  </div>
  <div style="flex: 1; text-align: center;">
    <strong>With Anti-Aliasing</strong><br>
    <img src="./ch_06_antialiasing/sphere_world_cropped.jpg" alt="With Anti-Aliasing" style="width: 95%; height: auto;">
  </div>
</div>

## Diffuse Material
![Diffuse Material](./ch_07_diffuse_material/Diffuse_Material.jpg)

# Debugging Tools

* **compute-sanitizer** [path/to/cuda_executable]
* **cuda-gdb** [path/to/cuda_executable] + run [args]
* [**demangler**](http://demangler.com/) - just put GCC symbol name and get demangled name
