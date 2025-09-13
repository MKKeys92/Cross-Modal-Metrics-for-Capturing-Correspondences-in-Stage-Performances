# The Intention-Based Abstraction Layer for AI-Driven Light Show Generation

## 1. The Rationale for Abstraction in Creative AI

The direct application of machine learning models to raw lighting control data, such as DMX-512 streams, presents two fundamental and prohibitive challenges: the **curse of dimensionality** and the **semantic gap**. A single professional light show can generate millions of DMX parameter values per minute, creating a high-dimensional dataset that is computationally intractable for standard deep learning architectures. More critically, these raw values lack inherent artistic meaning; a DMX value of 255 on a specific channel does not, on its own, describe the creative intent of a "sharp, intense strobe" versus a "slow, gentle fade."

To overcome these limitations, this research introduces a crucial abstraction layer. This layer is designed not merely to reduce data, but to translate it into a language of artistic intent. It achieves this through two core principles: **semantic grouping** and **feature extraction**.

### The Principle of Semantic Grouping
First, individual luminaires are clustered into semantically coherent units based on their physical position (e.g., front truss, back truss, stage floor) and aesthetic function (e.g., key light, backlight, effects). This process mirrors the workflow of a human lighting designer, who selects a group of fixtures to create a unified look, such as a full-stage wash. This strategy reduces the output dimensionality from hundreds of individual fixtures to a manageable set of semantic groups (typically 8-16), allowing the model to learn artistically relevant relationships.

### The Benefits of Abstraction
This approach yields several critical benefits that make AI-driven generation feasible:

*   **Achieving a Fixture-Agnostic Model**: By abstracting away the specifics of hardware, the model learns the *what* (the artistic intent, like a rhythmic pulse) rather than the *how* (the specific DMX values for a particular fixture). This makes the trained model universally applicable. The generated abstract output can be translated back to *any* lighting rig through a separate interpretation step, a critical requirement for real-world usability.
*   **Bridging the Semantic Gap**: The layer transforms raw, numerical data into features that represent meaningful artistic concepts like peak intensity, color, and dynamic range. This provides the model with a representation that is directly correlated with creative choices.
*   **Enabling Learnability**: This process provides a necessary **inductive bias**, guiding the model toward learning musically and visually coherent patterns. The abstracted features create a cleaner, more stable signal, focusing the learning process on the meaningful relationships between sound and light.

---

## 2. The Abstracted Features and Their Formulation (Global Intention-Based)

The Global Intention-Based model describes a lighting show as a set of continuous feature curves for each defined group of luminaires. These curves represent the collective behavior and artistic intent for that group over time. The following key parameters are extracted:

**Intensity of Peak absolute ($I_{\text{peak}}$)**: Represents the maximum brightness of any single fixture within a group at a given moment. This feature captures the absolute peak energy output of the group, essential for identifying accents and moments of high intensity.
  
$$
I_{\mathrm{peak}}(g,t) = \max_{i \in g} I_i(t)
$$
    
where $g$ is the set of luminaires in the group and $I_i(t)$ is the intensity of an individual luminaire.

**Spatial Intensity Gradient ($\nabla_S I$)**: Measures the sharpness of the intensity distribution *across the fixtures within a group* at a single frame. This feature is crucial for distinguishing between a stark, high-contrast look where a single fixture is highlighted, and a smooth, cohesive wash of light. A high gradient value indicates a sharp falloff in brightness between adjacent fixtures (a "spiky" look), while a low value signifies a smooth intensity distribution across the group (a "wash" or "fan"). This is a spatial metric, not a measure of temporal change.
  
$$
\nabla_S I(g, t) = \frac{1}{N_{\text{fixtures}} - 1} \sum_{i=2}^{N_{\text{fixtures}}} |I_i(t) - I_{i-1}(t)|
$$
    
where fixtures $i$ in group $g$ are spatially ordered, and $I_i(t)$ is the intensity of an individual fixture.

**AF Peak Density ($\rho_{\text{peak}}$)**: An "Alternating Factor" that quantifies the spatial complexity of the intensity distribution across the group at a single frame. A low value indicates a simple look with one or few dominant light sources, while a high value signifies a complex pattern with many alternating points of light.

$$
\rho_{\text{peak}}(g, t) = \frac{N_{\text{peaks}}(g, t)}{N_{\text{fixtures}}}
$$

where $N_{\text{peaks}}$ is the number of distinct spatial intensity peaks across the fixtures in group $g$ at time $t$.

**Contrast Index (inverse-minimum intensity, C)** This metric captures the contrast or dynamic range within the group. By inverting the minimum intensity value, a score near \(1.0\) indicates high contrast (at least one fixture is near zero intensity), while a score near \(0\) indicates low contrast (all fixtures are illuminated).

$$
C(g,t) = 1 - \min_{i \in g} I(i,t)
$$

where \(I(i,t)\) is the intensity of fixture \(i\) at time \(t\), and \(g\) is the set of luminaires in the group.



**Color Hue Mean ($\bar{H}$)**: The average hue of all fixtures in the group. This represents the dominant color of the group's output, providing a single value for the overall color aesthetic.
  
$$
\bar{H}(g, t) = \text{mean}_{i \in g}(H_i(t))
$$

**Color Saturation Mean ($\bar{S}$)**: The average saturation of all fixtures in the group. This feature describes the purity or vividness of the dominant color, distinguishing between pastel shades and deeply saturated tones.
  
$$
\bar{S}(g, t) = \text{mean}_{i \in g}(S_i(t))
$$

---

## 3. The Journey of Data: A Practical Example

To illustrate this transformation, we will examine a 300-frame (10-second) segment from a performance, focusing on a specific group of fixtures.

### Step 1: The Raw Data Input

The process begins with the raw, preprocessed DMX data, where each fixture's parameters are available as a time-series array. Figure 1 shows the `BRIGHTNESS` values for several individual fixtures within the group. This representation is noisy, complex, and highly dimensional, making it a poor direct input for a machine learning model.

![Raw Brightness Plot](Figure_CustomRaw.png)
*Figure 1: The raw brightness values for individual fixtures. The high-frequency detail and disparate behavior make it difficult for an AI to discern a single, coherent artistic intent.*

### Step 2: The Transformation Process

The raw parameter data for all fixtures in the group is fed into the abstraction algorithms. The multiple streams of brightness, hue, and saturation are aggregated and transformed into a single set of six feature curves for that group, representing its collective behavior.

### Step 3: The Abstracted Output

The output of this process is shown in Figure 2. The chaotic individual signals have been transformed into a smooth, low-dimensional, and semantically rich representation of the group's behavior.

![Abstracted Data Plot](Figure_AbstractedData.png)
*Figure 2: The six global intention-based features for the entire group. The raw data has been transformed into a learnable representation capturing artistic intent, including dynamic plots for intensity-related features and visualizations for the average group color.*

---
