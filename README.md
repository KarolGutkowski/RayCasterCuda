# CUDA Ray Casting
<p>This is my 2nd project for CUDA computing class.</p>
<p>This project is my implementation of ray casting of spheres.</p>
<p>The rendered scene is processed on GPU using CUDA API and CUDA OpenGL interop.</p>
<p>The rendering was optimized using BVH algorithm which allows to quickly determine which parts of the scene should be processed by which threads.</p>
<p>The performance of my solution was measured on RTX 2060 and I've been able to average above 100fps for scene with 1000 spheres</p>
<p>For contrast I've also developed a similar CPU algorithm which wasn't able to get anywhere close to the GPU speeds</p>

![image](https://github.com/KarolGutkowski/RayCasterCuda/assets/90787864/a9ef10ba-1548-4fde-9c34-1369a09ca3e0)
10 000 spheres rendered in real-time (at around 50fps)
