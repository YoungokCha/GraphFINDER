## Graph-FINDER: Enabling Data-Driven Dopant Discovery in Phase Change Materials via Automated Text and Figure Extraction

### Explore the data extracted by Graph-FINDER via the VO₂ dopant database interactive web interface. ➜ [https://github.com/<owner>/VO2-dopant-database](https://wm355.github.io/GraphFinderDatabase/)



<p align="center">
 <img width="600" height="400" display="center" alt="fig_intro_overview" src="https://github.com/user-attachments/assets/a67a79e6-1033-4d24-9a30-5119747fe35e" />
</p>

Graph-FINDER is a multimodal AI framework links literature mining with machine learning–based prediction to establish a scalable foundation for identifying optimal material candidates. In the database creation stage (blue), Graph-FINDER employs natural language processing, computer vision, and large language models to extract textual and graphical information from publications, converting fragmented literature into curated, machine-actionable datasets. These datasets enable application-aware prediction (orange), where models estimate candidate properties, benchmark them against device requirements, and uncover interpretable structure–property relationships. The synthesis and validation stage (green) is included as a prospective extension rather than a contribution of this work, representing future experimental realization and measurement of top-ranked candidates to refine the database. This envisioned feedback loop highlights the broader potential of Graph-FINDER for accelerating functional materials discovery.


1. crop_roi_from_graph.py: Isolating curves from published figures, including region detection, axis localization, and text annotation
<p align="center">
 <img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/12572a36-00fa-498d-b897-9a9a3ad85e45" />
</p>


The process begins with acquiring original images from literature sources (a). Visual feature detection is then used to identify regions likely containing closed rectangles (b). The main plotting areas, typically defined by rectangular regions exceeding a specific area threshold and including the primary axes, are localized and extracted to focus on the core measurement content (c). Subplots and auxiliary axes are removed to reduce visual clutter, ensuring only the relevant data region is retained (d). Finally, textual elements such as legends, labels, and annotations are detected to preserve essential contextual information needed for accurate interpretation of the extracted data (e).

```python
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.007 * cv2.arcLength(contour, True), True)
    if len(approx) > 2:
        area = cv2.contourArea(contour)
        if area > (image_area / max_main_counter):
            if not any(is_duplicate(approx, existing) for existing in main_rectangles) and not any(is_inside(approx, existing) for existing in main_rectangles):
                main_rectangles.append(approx)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 10)
            else:
                detected_rectangles.append(approx)

main_counter = len(main_rectangles)
```


2. save_data_point_into_file.py: Reconstruction of numerical data from the graphs
<p align="center">
 <img width="12752" height="4488" alt="2FigureValGraph" src="https://github.com/user-attachments/assets/d48cc632-0160-4084-88f5-2041834afb95" />
</p>


 The process begins with a sample graph, where individual curves are separated through binary color masking. A virtual grid is then overlaid to detect and isolate data points corresponding to each curve. The extracted values are digitized into structured numerical tables, which are subsequently plotted to verify accuracy. Finally, the reconstructed plots are combined to reproduce the original figure with high precision.

```python
# Convert filtered points to arrays
    filtered_points = np.array(filtered_points)
  
    for x_value, y_value in filtered_points:
        cell_locations.append([legend, x_value, y_value])
        col = int((x_value - x_min) / x_scale)
        row = int((y_max - y_value) / y_scale)
        img[row - 2:row + 2, col - 2:col + 2] = 200
        
        if y_max < 10:
            x_value_file = x_value
            y_value_file = 10**(y_value)
        else:    
            x_value_file = x_value 
            y_value_file = y_value

        cell_locations_file.append([legend, x_value_file, y_value_file])
    
    plt.imshow(img, cmap='gist_ncar')
    for i in range(1, num_cols):
        plt.axvline(cols // num_cols * i, color='g', linestyle='--', linewidth=0.5)
    for i in range(1, num_rows):
        plt.axhline(rows // num_rows * i, color='g', linestyle='--', linewidth=0.5)
```

3. new_dopants_generation.py: Dopant–Property Prediction
<p align="center">
 <img width="800" height="400" alt="fig_method_prediction" src="https://github.com/user-attachments/assets/7eefa6ff-4e2b-48b6-9012-ccd74444c40b" />
</p>


A description generator utilized the Graph-FINDER database in combination with the Mendeleev AP to extract detailed chemical and physical descriptors of VO2 samples doped with known elements (e.g., molybdenum). These descriptors include information such as atomic number, oxidation state, ionic radius, and electronegativity. The generated textual descriptions are then transformed into numerical text embeddings using OpenAI’s language model and subsequently fed into a multi-layer perceptron (MLP) regressor. The MLP is trained to predict key metal–insulator transition (MIT) parameters, including the resistance ratio, hysteresis width and transition temperature. Once trained, the MLP regressor is used to evaluate potential candidate dopants. Descriptions of these candidates are processed in the same way as the training data, enabling the model to predict their corresponding MIT properties.

```python

    # Add cross-validation for more robust evaluation
    cv_metrics = {'mse': [], 'r2': []}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        cv_model = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation='tanh',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
        cv_model.fit(X_train, Y_train)
        Y_val_pred = cv_model.predict(X_val)

        cv_metrics['mse'].append(mean_squared_error(Y_val, Y_val_pred))
        cv_metrics['r2'].append(r2_score(Y_val, Y_val_pred))

    avg_metrics['cv_mse'] = np.mean(cv_metrics['mse'])
    avg_metrics['cv_r2'] = np.mean(cv_metrics['r2'])

```
