# _*_ coding:utf-8 _*_

def f(x):
    return x * x * x - 12 * x - 11

def main():
    a = 0
    b = 10
    e = 0.01

    x1 = a + 0.382 * (b - a)
    x2 = a + b - x1
    f1 = f(x1)
    f2 = f(x2)

    while True:
        print('a: %f____b: %f' % (a, b))
        if f1 > f2:
            a = x1
            if (b - a) < e:
                break
            f1 = f2
            x1 = x2
            x2 = a + b - x1
            f2 = f(x2)
        else:
            b = x2
            if (b - a) < e:
                break
            f2 = f1
            x2 = x1
            x1 = a + b - x2
            f1 = f(x1)

    print('x: %f   f(x)= %f' % ((a + b) / 2, f((a + b) / 2)))

if __name__ == '__main__':
    main()