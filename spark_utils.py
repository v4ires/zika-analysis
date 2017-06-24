def list2csv(arr_input):
    '''Method for converting python list into csv file'''
    output = io.StringIO("")
    csv.writer(output).writerow(arr_input)
    return output.getValue().strip()

def csv2rdd(csv):
    '''Method for converting csv file into rdd'''
    return sc.parallelize(csv)

def rdd2numpy_arr(rdd_input):
    '''Method for converting rdd into numpy array'''
    return np.asarray(rdd.collect())

def save_result(dir_path, rdd_input):
    '''Method for save rdd result into specific folder'''
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    rdd_input.saveAsTextFile(dir_path)

def kmeans_error(point, clusters_rdd): 
    '''Method for evaluate the kmeans algorithm error'''
    center = clusters_rdd.centers[clusters_rdd.predict(point)] 
    return sqrt(sum([x**2 for x in (point - center)]))    

def show_rdd(x):
    '''Method for show the RDD content'''
    print(x)
