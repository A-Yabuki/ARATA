"""
FCN (Encoder -> Part of Decorder (until size 1/4))
    (Divided into S * S, st. each window size = (Hi / S, Wi / S))
    -> Category ( S * S * Ch)
    -> Mask (H * W * Ch + 2)
        (CoordConv)... add two channels
            1. x coods (normalized -1~1)
            2. y coods (normalized -1~1)

       (*7 Conv)
    -> Category (S * S * Ch)
    -> Mask (H * W * Ch)

Mask Output Hi * Wi * S**2
( 'k'th channel = segment instance at grid(i, j) 
   k= i * (S + j))


"""
