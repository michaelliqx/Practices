
def SpiralMatrix(matrix):
    res = []
    if not matrix:
        return []
    row, col = len(matrix),len(matrix[0])
    print(row,col)
    i,j= 0,0
    while row>1 and col>1:

        for k in range(col-1):
            res.append(matrix[i][j])
            j+=1
        for k in range(row-1):
            res.append(matrix[i][j])
            i+=1
        for k in range(col-1):
            res.append(matrix[i][j])
            j-=1
        for k in range(row-1):
            res.append(matrix[i][j])
            i-=1
        row-=2
        col-=2
    i+=1
    j+=1
    if col == 1:
        for k in range(row):
            res.append(matrix[i][j])
            i += 1
    elif row == 1:
        for k in range(col):
            res.append(matrix[i][j])
            j += 1


    return res


def main():
    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    a = SpiralMatrix(matrix)
    print(a)

if __name__ == '__main__':
    main()