import os
import cv2
import numpy as np

def generate_synthetic_matting_video(output_path="input/sample_matting_video.mp4", num_frames=180, width=800, height=600, fps=30):
    """
    Generates a synthetic video showing a person moving in front of a cluttered room background 
    to demonstrate AI alpha matting and virtual background replacement.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    print(f"[+] Generating synthetic matting video: '{output_path}' ({num_frames} frames)...")
    
    # Background: Cluttered Office Scene (Bookshelf + Window + Wall)
    bg = np.ones((height, width, 3), dtype=np.uint8) * 50
    # Bookshelf on left
    cv2.rectangle(bg, (20, 80), (220, 520), (35, 65, 110), -1)
    for y in range(120, 500, 70):
        cv2.line(bg, (20, y), (220, y), (20, 45, 80), 3)
        cv2.rectangle(bg, (40, y - 40), (90, y), (0, 180, 200), -1)
        cv2.rectangle(bg, (100, y - 45), (150, y), (200, 80, 0), -1)
        
    # Window on right
    cv2.rectangle(bg, (width - 240, 60), (width - 40, 360), (220, 190, 120), -1)
    cv2.rectangle(bg, (width - 240, 60), (width - 40, 360), (50, 50, 50), 3)
    cv2.line(bg, (width - 140, 60), (width - 140, 360), (50, 50, 50), 2)
    
    for f in range(num_frames):
        frame = bg.copy()
        
        # Smooth person movement (left to right)
        shift_x = int(60 * np.sin(f / 15.0))
        cx = width // 2 + shift_x
        cy = height // 2 + 50
        
        # Draw Person Foreground (Subject)
        # Head
        cv2.circle(frame, (cx, cy - 140), 65, (180, 210, 240), -1)
        cv2.circle(frame, (cx, cy - 140), 65, (80, 100, 120), 2)
        # Hair
        cv2.ellipse(frame, (cx, cy - 165), (70, 45), 0, 180, 360, (30, 30, 30), -1)
        # Torso & Shoulders
        body_pts = np.array([
            [cx - 160, height],
            [cx - 110, cy - 60],
            [cx + 110, cy - 60],
            [cx + 160, height]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [body_pts], (200, 80, 40)) # Coral red shirt
        
        # Frame Text
        cv2.putText(frame, f"AI MATTING TEST BENCH | FRAME: {f+1:03d}/{num_frames}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)
                    
        out.write(frame)
        
    out.release()
    print(f"[OK] Synthetic matting video saved to '{output_path}'")
    return output_path

if __name__ == "__main__":
    generate_synthetic_matting_video()
