import streamlit as st
import cv2
import numpy as np
import pandas as pd


st.title("🎓 Deep Learning OMR Grader")
st.write("Upload your OMR sheet and answer key CSV file")

# File uploaders
uploaded_img = st.file_uploader("Upload OMR Image", type=["jpg", "jpeg", "png"])
uploaded_key = st.file_uploader("Upload Answer Key CSV", type=["csv"])

st.info("📋 Answer key format: `question,answer` (1-indexed, e.g., A=1, B=2, C=3, D=4, E=5)")

# -------------------------
# Load answer key - FIXED VERSION
# -------------------------
def load_answer_key(file_like):
    """Load answer key from CSV file"""
    df = pd.read_csv(file_like)
    df.columns = df.columns.str.strip()
    
    answer_dict = {}
    for _, row in df.iterrows():
        q = int(row["question"])
        ans = int(row["answer"]) - 1  # Convert to 0-indexed
        answer_dict[q] = ans
    
    return answer_dict

# -------------------------
# Bubble classifier using image analysis (no TensorFlow needed)
# -------------------------
def classify_bubble_cv(gray_img, bubble, size=40):
    """Classify a bubble as filled or empty using image analysis.
    Returns a confidence score between 0 (empty) and 1 (filled)."""
    x, y, r = bubble['x'], bubble['y'], bubble['r']
    
    # Extract region
    pad = int(r * 1.3)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(gray_img.shape[1], x + pad), min(gray_img.shape[0], y + pad)
    
    crop = gray_img[y1:y2, x1:x2]
    
    if crop.size == 0:
        return 0.0
    
    # Method 1: Darkness score (lower pixel value = darker = filled)
    center_region = crop[crop.shape[0]//4:3*crop.shape[0]//4, 
                         crop.shape[1]//4:3*crop.shape[1]//4]
    if center_region.size == 0:
        return 0.0
    darkness = (255 - np.mean(center_region)) / 255.0
    
    # Method 2: Threshold-based fill ratio
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Create circular mask
    mask = np.zeros_like(binary)
    center = (crop.shape[1] // 2, crop.shape[0] // 2)
    radius = min(crop.shape[0], crop.shape[1]) // 3
    cv2.circle(mask, center, radius, 255, -1)
    # Calculate fill ratio within the circle
    masked_binary = cv2.bitwise_and(binary, mask)
    circle_area = np.sum(mask > 0)
    filled_area = np.sum(masked_binary > 0)
    fill_ratio = filled_area / circle_area if circle_area > 0 else 0
    
    # Method 3: Standard deviation (filled bubbles have lower std in center)
    std_score = 1.0 - min(1.0, np.std(center_region) / 80.0)
    
    # Combine scores
    combined_score = darkness * 0.4 + fill_ratio * 0.45 + std_score * 0.15
    
    return combined_score

# -------------------------
# Enhanced preprocessing
# -------------------------
def preprocess_image(img_bgr):
    h, w = img_bgr.shape[:2]
    if h > 1500:
        scale = 1500 / h
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return img_bgr, gray

# -------------------------
# Improved bubble detection using multiple methods
# -------------------------
def detect_bubbles_hough(gray, debug=False):
    """Detect bubbles using Hough Circle Transform with better parameters"""
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Try multiple parameter sets to catch all bubbles
    all_circles = []
    
    # Parameter set 1 - smaller bubbles
    circles1 = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=20,
        minRadius=10,
        maxRadius=25
    )
    if circles1 is not None:
        all_circles.extend(circles1[0])
    
    # Parameter set 2 - medium bubbles
    circles2 = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=40,
        param2=25,
        minRadius=12,
        maxRadius=22
    )
    if circles2 is not None:
        all_circles.extend(circles2[0])
    
    # Remove duplicates
    bubbles = []
    for circle in all_circles:
        x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
        
        # Check if this is a duplicate
        is_duplicate = False
        for existing in bubbles:
            dist = np.sqrt((x - existing['x'])**2 + (y - existing['y'])**2)
            if dist < 15:  # Too close to existing
                is_duplicate = True
                break
        
        if not is_duplicate:
            bubbles.append({
                'x': x,
                'y': y,
                'r': r,
                'area': np.pi * r * r
            })
    
    # Filter out bubbles in top 15% of image (header area with "OMR ANSWER SHEET")
    h = gray.shape[0]
    header_cutoff = int(h * 0.15)
    bubbles = [b for b in bubbles if b['y'] > header_cutoff]
    
    # Sort by position
    bubbles = sorted(bubbles, key=lambda b: (b['y'], b['x']))
    
    if debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Draw header cutoff line
        cv2.line(vis, (0, header_cutoff), (vis.shape[1], header_cutoff), (255, 0, 0), 2)
        cv2.putText(vis, "Header area (ignored)", (10, header_cutoff - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        for b in bubbles:
            cv2.circle(vis, (b['x'], b['y']), b['r'], (0, 255, 0), 2)
            cv2.circle(vis, (b['x'], b['y']), 2, (0, 0, 255), 3)
            # Show coordinates
            cv2.putText(vis, f"r={b['r']}", (b['x']-15, b['y']-b['r']-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        st.image(vis, caption=f"Detected {len(bubbles)} circles (excluding header)", use_container_width=True)
    
    return bubbles

# -------------------------
# Organize into questions with better logic
# -------------------------
def organize_into_questions(bubbles, n_choices=5):
    """Group bubbles into rows based on y-coordinate clustering"""
    if not bubbles:
        return []
    
    # Sort by y-coordinate
    sorted_bubbles = sorted(bubbles, key=lambda b: b['y'])
    
    # Cluster into rows using adaptive threshold
    rows = []
    current_row = [sorted_bubbles[0]]
    
    # Calculate average radius to determine row threshold
    avg_radius = np.mean([b['r'] for b in bubbles])
    y_threshold = int(avg_radius * 1.5)  # Adaptive threshold
    
    for bubble in sorted_bubbles[1:]:
        if abs(bubble['y'] - current_row[-1]['y']) <= y_threshold:
            current_row.append(bubble)
        else:
            # Save current row if it has enough bubbles
            if len(current_row) >= n_choices:
                sorted_row = sorted(current_row, key=lambda b: b['x'])[:n_choices]
                rows.append(sorted_row)
            elif len(current_row) == n_choices - 1:  # Allow 4 bubbles (might be missing one)
                sorted_row = sorted(current_row, key=lambda b: b['x'])
                rows.append(sorted_row)
            current_row = [bubble]
    
    # Add last row
    if len(current_row) >= n_choices - 1:
        sorted_row = sorted(current_row, key=lambda b: b['x'])
        if len(sorted_row) >= n_choices:
            sorted_row = sorted_row[:n_choices]
        rows.append(sorted_row)
    
    return rows

# -------------------------
# Main processing
# -------------------------
if uploaded_img and uploaded_key:
    # Load answer key
    try:
        answer_key = load_answer_key(uploaded_key)
        total_questions = len(answer_key)
        st.success(f"✅ Loaded answer key with {total_questions} questions")
        
        with st.expander("📋 View Answer Key"):
            preview_data = []
            for q in sorted(answer_key.keys()):
                ans_idx = answer_key[q]
                ans_letter = chr(65 + ans_idx)
                preview_data.append({
                    'Question': q,
                    'Correct Answer': f"{ans_letter} ({ans_idx + 1})"
                })
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error loading answer key: {str(e)}")
        st.stop()
    
    # Load image
    file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        st.error("❌ Could not read image")
        st.stop()
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded OMR Sheet", use_container_width=True)
    
    # Process
    with st.spinner("🔍 Processing image..."):
        img_processed, gray = preprocess_image(img_bgr)
        
        # Display grayscale image
        st.image(gray, caption="Grayscale Preprocessed Image", use_container_width=True, channels="GRAY")
        
        # Show binary threshold and edge detection for explanation
        col1, col2 = st.columns(2)
        with col1:
            # Binary threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            st.image(binary, caption="Binary Threshold Image", use_container_width=True, channels="GRAY")
        with col2:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            st.image(edges, caption="Edge Detection (Canny)", use_container_width=True, channels="GRAY")
        
        # Detect bubbles
        bubbles = detect_bubbles_hough(gray, debug=True)
        st.write(f"🔵 Detected {len(bubbles)} bubbles")
        
        if len(bubbles) < total_questions * 3:
            st.warning(f"⚠️ Expected ~{total_questions * 5} bubbles but found {len(bubbles)}")
        
        # Organize into questions
        questions = organize_into_questions(bubbles, n_choices=5)
        st.write(f"📝 Organized into {len(questions)} questions")
        
        if len(questions) == 0:
            st.error("❌ Could not organize bubbles into rows")
            st.stop()
    
    # Grade
    st.write("---")
    st.subheader("📊 Grading Results")
    
    results = []
    correct_count = 0
    
    for q_num, question_bubbles in enumerate(questions, start=1):
        if q_num > total_questions:
            break
        
        # Get scores for each bubble using CV-based classifier
        scores = []
        for bubble in question_bubbles:
            score = classify_bubble_cv(gray, bubble)
            scores.append(score)
        
        # Find filled bubble (highest score)
        max_score = max(scores)
        selected_idx = int(np.argmax(scores))
        
        # Threshold for determining if nothing is filled
        FILL_THRESHOLD = 0.40  # Adjusted threshold
        
        # Check if any bubble is actually filled
        if max_score < FILL_THRESHOLD:
            selected_letter = "NA"
            selected_conf = 0.0
            is_correct = False  # NA is always marked wrong
        else:
            selected_idx = int(np.argmax(scores))
            selected_conf = scores[selected_idx]
            
            # Get correct answer
            correct_idx = answer_key.get(q_num, None)
            
            # Check correctness
            is_correct = (selected_idx == correct_idx) if correct_idx is not None else False
            
            # Convert to letter
            selected_letter = chr(65 + selected_idx)
        
        if is_correct:
            correct_count += 1
        
        # Get correct answer letter
        correct_idx = answer_key.get(q_num, None)
        correct_letter = chr(65 + correct_idx) if correct_idx is not None else "N/A"
        
        results.append({
            'Question': q_num,
            'Selected': selected_letter,
            'Confidence': f"{max_score:.2f}",
            'Correct': correct_letter,
            'Status': '✅' if is_correct else ('⚠️' if selected_letter == "NA" else '❌')
        })
    
    # Display results
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    
    # Final score
    score_pct = (correct_count / total_questions * 100) if total_questions > 0 else 0
    st.write("---")
    st.success(f"🎯 **Final Score: {correct_count}/{total_questions} ({score_pct:.1f}%)**")
    
    # Visualize
    vis_img = img_processed.copy()
    for q_num, question_bubbles in enumerate(questions[:total_questions], start=1):
        for i, bubble in enumerate(question_bubbles):
            x, y, r = bubble['x'], bubble['y'], bubble['r']
            is_correct_ans = (i == answer_key.get(q_num, -1))
            color = (0, 255, 0) if is_correct_ans else (200, 200, 200)
            thickness = 3 if is_correct_ans else 1
            cv2.circle(vis_img, (x, y), r, color, thickness)
            cv2.putText(vis_img, chr(65 + i), (x - 8, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), 
             caption="Detected Bubbles (Green = Correct Answer)", 
             use_container_width=True)

else:
    st.info("👆 Please upload both an OMR image and answer key CSV to begin")
    
    with st.expander("📄 Sample Answer Key Format"):
        st.code("""question,answer
1,2
2,1
3,4
4,3
5,2""", language="csv")
        st.caption("Use 1=A, 2=B, 3=C, 4=D, 5=E")