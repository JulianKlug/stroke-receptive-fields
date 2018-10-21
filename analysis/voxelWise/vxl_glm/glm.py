# eval : this will have to be done at model level (otherwise it will use to much RAM)
    # # Initialising arrays that will contain all data
    # all_subj_X_train = np.empty([np.sum(balancing_selector), input_size])
    # all_subj_y_train = np.empty(np.sum(balancing_selector))
    # all_subj_index = 0
    #
    #         all_subj_X_train[all_subj_index : all_subj_index + subj_X_train.shape[0], :] = subj_X_train
    #         all_subj_y_train[all_subj_index : all_subj_index + subj_y_train.shape[0]] = subj_y_train
    #         all_subj_index += subj_X_train.shape[0]
