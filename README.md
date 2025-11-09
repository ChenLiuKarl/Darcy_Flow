# 2D Darcy Flow PDE Learning

This project contains two neural network models designed to **learn the solution operator of the 2D Darcy flow problem**, where the goal is to predict the pressure field \( u(x) \) given the permeability field \( a(x) \), with the source term \( f(x) \) as a constant.

---

## 🧠 Overview

The governing equation for the **2D Darcy Flow** problem on the unit box is:

\[
-\nabla \cdot \left( a(x) \nabla u(x) \right) = f(x), \quad x \in (0,1)^2,
\]

subject to the **Dirichlet boundary condition**:

\[
u(x) = 0, \quad x \in \partial(0,1)^2.
\]

To approximate the solution operator of this PDE, two neural network architectures are implemented and compared:

1. **Convolutional Neural Network (CNN)**  
2. **Resolution-Invariant Fourier Neural Operator (FNO)**

While the CNN achieves good results after about **200 epochs of training**, the FNO reaches **comparable accuracy in only 10 epochs**. The test error keeps decreasing significantly with further training.  
Furthermore, the FNO is **resolution invariant**, meaning the trained model maintains performance even when tested on higher-resolution inputs.

---

## ⚙️ Models

### 1. Convolutional Neural Network (CNN)
The **CNN** treats the permeability field \( a(x, y) \) and pressure field \( u(x, y) \) as single-channel 2D images and uses standard convolutional layers for learning.

**Key features:**
- Encoder–bottleneck–decoder architecture  
- Optimized with **Adam** optimizer  
- Includes a **learning rate (LR) scheduler** for stability  
- Learns a mapping from the input field \( a(x, y) \) to the output field \( u(x, y) \)

---

### 2. Fourier Neural Operator (FNO)
The **FNO** leverages spectral (Fourier) transformations to learn mappings between function spaces directly.

**Key features:**
- Inputs: spatial coordinates \((x, y)\) and permeability field \(a(x, y)\) — total **3 input channels**  
- Data lifted to **32 channels** before passing through 2D Fourier layers  
- Uses **first 12 positive and negative frequency modes** in spectral space  
- Outputs a **single pressure field** \( u(x, y) \)  
- Optimized with **Adam** optimizer and **LR scheduler**  
- **Resolution invariant** — the model generalizes across grid resolutions by operating in the frequency domain

---

## 📈 Performance Summary

| Model | Epochs to Converge | Error (Relative) | Resolution Invariance |
|--------|-------------------|------------------|-----------------------|
| CNN    | ~200              | Moderate         | ❌ No                 |
| FNO    | ~10               | Small            | ✅ Yes                |

---

## 🙏 Acknowledgments

Special thanks to **Dr. Burigede Liu** and **Ms. Rui Wu** for providing the initial code framework as part of **Course 4C11 at CUED**.
