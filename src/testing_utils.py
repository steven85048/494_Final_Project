
def print_class_count( y ):
    # Obtain the number of zeroes and ones
    num_zero = 0
    num_one = 0
    for val in y:
        if( val == 0 ):
            num_zero += 1
        else:
            num_one += 1

    print( "The number of zeroes and ones is: " )
    print( str(num_zero) + " " + str(num_one))


def pca_plot_2d( X, y ):   
    pca = PCA( n_components = 2 )
    pcaX = pca.fit_transform(X)

    # may want to reduce the size of this
    PCA_dataframe = pd.DataFrame({"y": y[:,0], "R1": pcaX[:,0], "R2": pcaX[:,1]})

    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    labels = ['R1', 'R2']

    for line in dataFrame.itertuples():
        classIndex = 0 if line.y == 0 else 1

        plt.scatter(
            line.R1,
            line.R2,
            c=colors[classIndex], label=labels[classIndex], marker=markers[classIndex]
        )

    plt.title("2D PCA")
    plt.legend(loc='upper right')
    plt.show()