{-# LANGUAGE DataKinds #-}

import Control.Category
import Control.Monad.Random
import Data.MyPrelude
import Data.Utils
import Data.List (cycle)
import Numeric.Neural
import Prelude hiding ((.))

type BitPair = (Bool, Bool)

rawSamples :: [] BitPair
rawSamples = [ (True,  False),
               (False, True),
               (False, False),
               (True,  True)
             ]
-- Problem here is raw sample (4 permutations for all possible XOR I/O) is too few, so accuracy never reaches > 0.75
-- (if one get wrong it drops to 0.75, while the first result is almost doomed to be wrong).

samples = take 17 $ cycle rawSamples

-- With tanhLayer and learning rate at const 0.5,
-- if I take only 4n (ex: 8, 16...), accuracy will only reach 0.75.
--
-- 8 : 0.75
-- 9 : 1.0
-- 10: 0.8
-- 11: 1.0
-- 12: 0.75
-- 13: 0.769
-- 14: 0.786
-- 15: 0.733
-- 16: 0.75
-- 17: 1.0 
-- 18: 0.778 
-- 19: 0.737 
-- 20: 0.75
--
-- 31: 0.742
-- 32: 0.75
-- 33: 0.758
-- 63: 0.747
-- 64: 0.75
-- 65: 0.758
-- 255: 0.749
-- 256: 0.75
-- 257: 0.7509

toDouble True  = 1.0
toDouble False = 0.0

toBool :: Double -> Bool
toBool d | d == 1.0 = True
         | d == 0.0 = False
         | otherwise = (round.abs) d == 1

xor :: BitPair -> Bool
xor (a, b) | a == b = False
           | otherwise = True

expandPair :: BitPair -> Vector 2 Double
expandPair (a, b) = cons (toDouble a) $ cons (toDouble b) nil

xorModel :: StdModel (Vector 2) (Vector 1) BitPair Bool
xorModel = mkStdModel
  ((tanhLayer :: Layer 2 2) >>> tanhLayer)
  (\x -> Diff $ Identity . sqDiff (pure $ fromDouble (toDouble x)))   -- While there is no explicit 'Double' in the signature, the value inside layers be inferred as Bool,
  expandPair                                                          -- even though this function will convert to Double from samples, as inputs of the input layer.
  (toBool.vhead)                                                      -- Convert to Bool as a real XOR function should do.

report ts = liftIO $ do
  printf "#%d ((True,True),%s) ((True,False),%s) @%f\n" gn tt tf qp
  where
    m = tsModel ts
    gn = tsGeneration ts 
    b2s True = "True"
    b2s False = "False"
    tt = b2s $ model m (True, True)
    tf = b2s $ model m (False, True)
    ck b c c' = if c == c' then 1 else 0
    qp = fromJust $ qualityP' m ck [(pair, xor pair) | pair <- samples] :: Double

check ts = return $ if gn == 1000 || qp >= 0.99 then Just () else Nothing
  where 
    gn = tsGeneration ts
    m = tsModel ts
    ck _ c c' = if c == c' then 1 else 0
    qp = fromJust $ qualityP' m ck [(pair, xor pair) | pair <- samples] :: Double

main :: IO ()
main = flip evalRandT (mkStdGen 902829) $ do
    m <- modelR $ whiten xorModel [pair | pair <- samples]
    runEffect $
      simpleBatchP [(pair, xor pair) | pair <- samples] 1
      >-> descentP m 1 (const 0.5)                          
      >-> reportTSP 10 report
      >-> consumeTSP check

