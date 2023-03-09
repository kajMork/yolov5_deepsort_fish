import json


def evaluate_frame_count(file_results):
    # Annotations file is not saved in the same format as the results file, so we need to index it differently
    anno_file = open('Annotations_full.json', 'r')
    json_file = open(file_results, 'r')
    # Get video names for indexing annotations file
    anno_data = json.load(anno_file)
    data = json.load(json_file)
    # Get annotations for video
    sum = 0
    negative_diff_count = 0
    positive_diff_count = 0
    wrong_count = []
    wrong_count_i = []
    i = 0
    count_seq_sum = 0
    for eval_video in data['results']:
        # Get ground truth for video
        try:
            anno_file = open('Annotations_full.json', 'r')
            anno_data = json.load(anno_file)
            gt_video = anno_data[eval_video['video']]

        except KeyError:
            print("Video not found in annotations")
            print("Assuming its Other_fish, so making a fake dict for it")
            anno_data = {
                eval_video['video']: {
                    "fish_count_frames": [],
                    'fish_count': 0,
                    'enter_frame': [],
                    'exit_frame': []
                }
            }
            gt_video = anno_data[eval_video['video']]

        # Get frame count for videos and compare
        sum += abs(gt_video['fish_count'] - eval_video['fish_count'])
        if abs(gt_video['fish_count'] - eval_video['fish_count']) > 0:
            if gt_video['fish_count'] - eval_video['fish_count'] < 0:
                negative_diff_count += 1
            else:
                positive_diff_count += 1
            wrong_count.append(eval_video['video'])
            wrong_count_i.append(i)

        if gt_video['fish_count_frames'] != eval_video['fish_count_frames']:
            count_seq_sum += 1
            print("fish_count_frames not equal for video: ", eval_video['video'], " ground truth sequence: ", gt_video['fish_count_frames'], " found sequence is: ", eval_video['fish_count_frames'])
        i += 1

    print('Average frame count difference: ' + str(sum / len(data['results'])))
    print('Number of videos with negative difference: ' + str(negative_diff_count))
    print('Number of videos with positive difference: ' + str(positive_diff_count))
    print('Percentage of videos counted right: ' + str(1 - (negative_diff_count + positive_diff_count) / len(data['results'])))
    print('Wrong sequence sum: ' + str(count_seq_sum))
    print('Videos with wrong count: ' + str(wrong_count))

    for video, i in zip(wrong_count, wrong_count_i):
        '''print(7 * '-' + video + 7 * '-')
        print('Ground truth total count : ' + str(anno_data[video]['fish_count']) + " vs " + str(data['results'][i]['fish_count']))
        print('Ground truth entering frame count : ' + str(anno_data[video]['enter_frame']) + " vs " + str(data['results'][i]['enter_frames']))
        print('Ground truth exiting frame count : ' + str(anno_data[video]['exit_frame']) + " vs " + str(data['results'][i]['exit_frames']))
        print('Measured sequence :' + str(data['results'][i]['sequence']))'''

    return str(1 - (negative_diff_count + positive_diff_count) / len(data['results']))
