from copy import deepcopy

import cv2
import numpy as np
import pandas as pd

import constants
from utils import measure_distance, convert_pixel_distance_to_meters, get_closest_player_by_point

player_stats_data = [{
    'frame_num': 0,
    'player_1_number_of_shots': 0,
    'player_1_total_shot_speed': 0,
    'player_1_last_shot_speed': 0,
    'player_1_total_player_speed': 0,
    'player_1_last_player_speed': 0,

    'player_2_number_of_shots': 0,
    'player_2_total_shot_speed': 0,
    'player_2_last_shot_speed': 0,
    'player_2_total_player_speed': 0,
    'player_2_last_player_speed': 0,
}]


def set_stats(player_mini_court_detections, ball_mini_court_detections, ball_shot_frames, mini_court):
    """
    :param ball_mini_court_detections: [{ball_id_1:[x1,y1,x2,y2]},{},{}...]
    :param ball_shot_frames: [11,58,95,131,182]
    :return:
    """
    print("set stats...")
    video_length = len(ball_mini_court_detections)
    for ball_shot_index in range(len(ball_shot_frames) - 1):  # last shot is not considered as there is no next shot
        # Get ball shot time
        start_frame = ball_shot_frames[ball_shot_index]
        end_frame = ball_shot_frames[ball_shot_index + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / constants.FPS

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           )
        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # player who hit the ball
        player_positions = player_mini_court_detections[start_frame]
        ball_position = ball_mini_court_detections[start_frame][1]
        player_shot_ball = get_closest_player_by_point(player_positions, ball_position)

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court()
                                                                               )

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(video_length))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Calc average speed
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / \
                                                          player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / \
                                                          player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / \
                                                            player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / \
                                                            player_stats_data_df['player_1_number_of_shots']

    return player_stats_data_df


def draw_stats(output_video_frames, player_stats):
    print("draw stats...")
    for index, row in player_stats.iterrows():
        # Get stats
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        # draw background
        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width = 350
        height = 230

        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 500
        end_x = start_x + width
        end_y = start_y + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        # put text
        text = "     Player 1     Player 2"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 80, start_y + 30),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 80),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 80),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 120),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 120),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "avg. S. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 160),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 160),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "avg. P. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 200),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 200),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return output_video_frames
