def label_validation_data_by_numbers(row):
    val = 0
    if row[0] == 0:
        val = 0
    elif row[0] > 0:
        if row[8] > 0.5:
            val = 5
        elif row[8] < -0.5:
            val = 6
        elif row[1] > 0.5 and row[6] == 0:
            val = 2
        elif row[1] < -0.5 and row[6] > 0:
            val = 3
        elif -0.5 < row[1] < 0.5:
            val = 4
        else:
            val = 1
    return val


def label_validation_data_by_strings(row):
    val = ''
    if row[0] == 0:
        val = 'Stop_Driving'
    elif row[0] > 0:
        if row[8] > 0.5:
            val = 'Lane_change_to_the_left'
        elif row[8] < -0.5:
            val = 'Lane_change_to_the_right'
        elif row[1] > 0.5 and row[6] == 0:
            val = 'Accelerate_in_the_lane'
        elif row[1] < -0.5 and row[6] > 0:
            val = 'decelerate_in_the_lane'
        elif -0.5 < row[1] < 0.5:
            val = 'Drive_with_constant_speed'
        else:
            val = 'Start_Driving'

    return val
