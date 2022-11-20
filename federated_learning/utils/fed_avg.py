def average_nn_parameters(parameters, size):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    #print(parameters)
    #temp_parameters = []
    #for e in parameters:
    #    if bool(e):
    #        temp_parameters.append(e)
    #parameters = temp_parameters
    #print(len(parameters))
    #print(len(temp_parameters))
    #for name in temp_parameters[0].keys():
    #    new_params[name] = sum([param[name].data for param in temp_parameters]) / size #len(parameters)
    #print("-------------------> params: ", parameters[0])
    for name in parameters[0].keys():
        #print("name", name)
        new_params[name] = sum([param[name].data for param in parameters]) / size

    return new_params
