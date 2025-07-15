import os
import cv2
import numpy as np
import yaml
import shutil
import scipy.io as sio
import sys
import argparse


def prepareVideos(AllTrials_path, video_folder, video_size=(224, 224)):
    """
    Process video trials for feature extraction.
    
    Parameters:
    -----------
    AllTrials_path : str
        Path to the Trial_data.mat file containing trial information
    video_folder : str
        Path to folder containing raw video files
    video_size : tuple, optional
        Target video size (width, height) for cropping (default: (224, 224))
        
    Returns:
    --------
    list
        List of processed video file paths
    """
    output_folder = "./sample"    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    config_path = os.path.join("configs", "s3d.yml")
    with open(config_path, "r") as f:
        s3d_config = yaml.safe_load(f)

    stack_size = s3d_config.get("stack_size")
    startPreDispOut = s3d_config.get("startPreDispOut")

    # Load trial data
    try:
        mat_data = sio.loadmat(AllTrials_path, squeeze_me=True, struct_as_record=False)
        AllTrials = mat_data['AllTrials']
        print(f"Loaded trial data from: {AllTrials_path}")
    except Exception as e:
        raise FileNotFoundError(f"Could not load trial data from {AllTrials_path}: {e}")
    
    processed_videos = []
    text_file = os.path.join(output_folder, "processed_video_paths.txt")
    
    # Clear existing text file
    if os.path.exists(text_file):
        os.remove(text_file)
    
    for trial in AllTrials:
        trial_num = trial.trial_num
        
        # Skip trials without stick_in_out_disp data
        if trial.info.stick_in_out_disp is None:
            print(f"Skipping trial {trial_num} due to missing stick_in_out_disp")
            continue
        
        # Find matching video file
        trial_num_str = f"{trial_num:03d}"
        matching_files = [f for f in os.listdir(video_folder) 
                         if f.endswith("cam-1.mp4") and f.split('_')[1] == trial_num_str]
        matching_file = matching_files[0]
        video_path = os.path.join(video_folder, matching_file)
        
        # Calculate frame boundaries
        stick_in_out_disp0 = int(trial.info.stick_in_out_disp[0]) - 1  # Convert to 0-based
        start_frame = max(0, stick_in_out_disp0 - startPreDispOut)
        end_frame = trial.beakTip.xyz.shape[0]
        
        
        # Open video and setup output
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue
            
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create cropped video
        cropped_filename = f"{os.path.splitext(matching_file)[0]}_cropped.mp4"
        cropped_path = os.path.join(output_folder, cropped_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(cropped_path, fourcc, input_fps, video_size)
        
        # Process video frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_written = 0
        
        for frame_idx in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, video_size)
            out.write(resized_frame)
            frames_written += 1
        
        # Add padding frames
        black_frame = np.zeros((video_size[1], video_size[0], 3), dtype=np.uint8)
        for _ in range(stack_size - 1):
            out.write(black_frame)
        
        cap.release()
        out.release()
        
        # Verify output
        cropped_cap = cv2.VideoCapture(cropped_path)
        cropped_cap.release()
        
        processed_videos.append(cropped_path)
        
        # Write to text file
        with open(text_file, 'a') as f:
            f.write(cropped_path + '\n')
    
    print(f"\nProcessing complete! Processed {len(processed_videos)} videos.")
    print(f"Video paths saved to: {text_file}")


def main():
    parser = argparse.ArgumentParser(description="Process video trials for feature extraction")
    parser.add_argument("AllTrials_path", help="Path to the Trial_data.mat file")
    parser.add_argument("video_folder", help="Path to folder containing raw video files")
    parser.add_argument("--video_size", nargs=2, type=int, default=[224, 224], metavar=("WIDTH", "HEIGHT"),
                        help="Target video size for cropping (default: 224 224)")
    args = parser.parse_args()

    video_size = tuple(args.video_size)
    prepareVideos(args.AllTrials_path, args.video_folder, video_size)


if __name__ == "__main__":
    main()
