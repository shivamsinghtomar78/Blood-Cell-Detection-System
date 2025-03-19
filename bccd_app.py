import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import time

 
st.set_page_config(
    page_title="Blood Cell Detection App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .main-header { font-size: 2.5rem; font-weight: 600; color: #4B0082; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; font-weight: 500; color: #6200EA; margin: 1rem 0; }
    .card { background: white; border-radius: 0.8rem; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    .stButton>button { background: #6200EA; color: white; border: none; border-radius: 0.5rem; padding: 0.5rem 1rem; }
    .metric-card { text-align: center; padding: 1rem; }
    .metric-value { font-size: 2rem; font-weight: 600; color: #6200EA; }
    .metric-label { font-size: 0.9rem; color: #555; }
    .cell-badge { padding: 0.2rem 0.5rem; border-radius: 1rem; font-size: 0.8rem; }
    .rbc-badge { background: rgba(255,82,82,0.2); color: #FF5252; border: 1px solid #FF5252; }
    .wbc-badge { background: rgba(33,150,243,0.2); color: #2196F3; border: 1px solid #2196F3; }
    .plt-badge { background: rgba(76,175,80,0.2); color: #4CAF50; border: 1px solid #4CAF50; }
    .footer { text-align: center; padding: 1rem; color: #777; font-size: 0.8rem; border-top: 1px solid #eee; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

 
st.markdown("<h1 class='main-header'>Blood Cell Detection System</h1>", unsafe_allow_html=True)

 
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "ℹ️ About", "❓ Help"])

 
with tab1:
    
    @st.cache_data
    def preprocess_image(uploaded_file):
        try:
            bytes_data = uploaded_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid image file")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None

    @st.cache_resource
    def load_model():
        try:
            model = YOLO("bccd_yolov10_best.pt")   
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    def perform_inference(model, image):
        try:
            results = model.predict(image, conf=0.25)
            return results[0]
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
            return None

    def create_visualization(image, results):
        if not results.boxes:
            return None
        fig = go.Figure()
        fig.add_trace(go.Image(z=image))
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        class_names = ["RBC", "WBC", "Platelets"]
        colors = ["#FF5252", "#2196F3", "#4CAF50"]
        for box, cls, conf in zip(boxes, classes, confs):
            x0, y0, x1, y1 = box
            cls_name = class_names[int(cls)]
            color = colors[int(cls)]
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=color, width=2))
            fig.add_annotation(x=x0, y=y0-5, text=f"{cls_name} ({conf:.2f})", showarrow=False, 
                               font=dict(color="white", size=10), bgcolor=color, bordercolor=color)
        fig.update_layout(width=700, height=500, margin=dict(l=0, r=0, b=0, t=0), showlegend=False)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        return fig

    def create_cell_counts_chart(results):
        if not results.boxes:
            return None
        classes = results.boxes.cls.cpu().numpy()
        class_names = ["RBC", "WBC", "Platelets"]
        counts = {name: sum(1 for cls in classes if class_names[int(cls)] == name) for name in class_names}
        df = pd.DataFrame({"Cell Type": list(counts.keys()), "Count": list(counts.values())})
        fig = px.bar(df, x="Cell Type", y="Count", color="Cell Type", 
                     color_discrete_sequence=["#FF5252", "#2196F3", "#4CAF50"], 
                     text=df["Count"], height=400)
        fig.update_layout(title="Cell Count Distribution", xaxis_title="", yaxis_title="Count")
        fig.update_traces(textposition="outside")
        return fig

    # Main Content
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    model = load_model()
    if not model:
        st.stop()

    st.markdown("<h2 class='sub-header'>Upload Image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            image = preprocess_image(uploaded_file)
            if image is None:
                st.stop()
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            if st.button("🔍 Detect Blood Cells"):
                with st.spinner("Analyzing..."):
                    time.sleep(1)  # Simulate processing
                    results = perform_inference(model, image)
                    if results and results.boxes:
                        vis_fig = create_visualization(image, results)
                        st.plotly_chart(vis_fig, use_container_width=True)
                        
                        # Metrics
                        classes = results.boxes.cls.cpu().numpy()
                        class_names = ["RBC", "WBC", "Platelets"]
                        counts = {name: sum(1 for cls in classes if class_names[int(cls)] == name) for name in class_names}
                        cols = st.columns(4)
                        cols[0].markdown(f"<div class='metric-card'><div class='metric-value'>{len(classes)}</div><div class='metric-label'>Total Cells</div></div>", unsafe_allow_html=True)
                        cols[1].markdown(f"<div class='metric-card'><div class='metric-value' style='color:#FF5252'>{counts['RBC']}</div><div class='metric-label'>RBC</div></div>", unsafe_allow_html=True)
                        cols[2].markdown(f"<div class='metric-card'><div class='metric-value' style='color:#2196F3'>{counts['WBC']}</div><div class='metric-label'>WBC</div></div>", unsafe_allow_html=True)
                        cols[3].markdown(f"<div class='metric-card'><div class='metric-value' style='color:#4CAF50'>{counts['Platelets']}</div><div class='metric-label'>Platelets</div></div>", unsafe_allow_html=True)
                        
                     
                        count_fig = create_cell_counts_chart(results)
                        st.plotly_chart(count_fig, use_container_width=True)
                        
                        
                        st.markdown("<h2 class='sub-header'>Detailed Results</h2>", unsafe_allow_html=True)
                        data = [{"ID": i+1, "Cell Type": f"<span class='cell-badge {['rbc','wbc','plt'][int(cls)]}-badge'>{class_names[int(cls)]}</span>", 
                                 "Confidence": f"{conf*100:.2f}%"} 
                                for i, (cls, conf) in enumerate(zip(classes, results.boxes.conf.cpu().numpy()))]
                        st.write(pd.DataFrame(data).to_html(escape=False, index=False), unsafe_allow_html=True)
                        
                  
                        st.download_button("📊 Export as CSV", 
                                          pd.DataFrame({"Cell Type": [class_names[int(cls)] for cls in classes], 
                                                        "Confidence": [f"{conf*100:.2f}%" for conf in results.boxes.conf.cpu().numpy()]}).to_csv(index=False), 
                                          "results.csv", "text/csv")
                    else:
                        st.warning("No cells detected or analysis failed.")
        st.markdown("</div>", unsafe_allow_html=True)

 
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>About</h2>", unsafe_allow_html=True)
    st.markdown("""
    This app uses YOLOv10 to detect Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets in blood smear images. 
    Upload an image in the Analysis tab to see counts, visualizations, and detailed results.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
 
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='sub-header'>Help</h2>", unsafe_allow_html=True)
    st.markdown("""
    - **Upload**: Select a JPG, JPEG, or PNG image in the Analysis tab.
    - **Detect**: Click "Detect Blood Cells" to analyze.
    - **Results**: View counts and export data as CSV.
    - **Support**: Email ss93134041@gmail.com for issues.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
