{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}

module Main where

import           Control.Applicative
import           Control.Arrow        hiding (loop)
import           Data.Attoparsec.Text
import qualified Data.Text            as T
import           Data.MyPrelude
import           Numeric.Neural
import           Data.Utils

main :: IO ()
main = do
    xs <- readSamples
    printf "read %d samples\n\n" (length xs)
    printf "generation  learning rate  model error  accuracy\n\n"
    (g, q) <- flip evalRandT (mkStdGen 123456) $ do
        m <- modelR (whiten irisModel $ fst <$> xs)
        runEffect $
                simpleBatchP xs 5
            >-> descentP m 1 (\i -> 0.1 * 5000 / (5000 + fromIntegral i))
            >-> reportTSP 1000 (report xs)
            >-> consumeTSP (check xs)
    printf "\nreached prediction accuracy of %5.3f after %d generations\n" q g

-- * the "dynamic learning rate function" is lower the learning rate while the accuracy is
-- getting higher and higher ("stepping down" with smaller steps while it's getting close?).
--
-- (I don't know if this is momentum from the definition but it can be)
--
-- "Momentum works by taking the gradient calculated by SGD and adding a factor to it.
--  The added factor can be thought of as the average of the previous gradients."
--
-- So it looks this function is actually taking the average of the previous gradients.
--
-- http://wiki.fast.ai/index.php/Dynamic_learning_rates
--
--
-- * the 'whiten' is for 'whitening transformation'.
-- 
-- From comment it briefly describes:
--
-- 1. Get data points of a specific shape
-- 2. Execute a function to normalize them
-- 3. Result: each component has 0 mean and unit variance
--
-- "component" here denotes a thing "behave like 'Arrow's with choice over a different base category",
-- which is used in a model to "measure their error with regard to samples and can be trained by gradient descent"
--
-- "layer" is a component as well.
--
-- The usage is:
--
--   Data -> Decorrelation -> Whitening (to make it fit a identity covariance matrix/shape)
--
-- In the decorrelation step data will be "orthogonal". The effect of that is now data points are "rotated"
-- by the eigenvector matrix from their covariance. It makes data align the principal axes that along data
-- has the largest variance. Then, whitening can "scale" them, so that data become centered in a sphere about the origin.
-- The result will be a clear sphere that filter out those values have the "same value" and make the true uniq values
-- can be pick up much easily.
--

  where

    report xs ts = liftIO $ 
        printf "%10d %14.4f %12.6f %9.4f\n" (tsGeneration ts) (tsEta ts) (modelError (tsModel ts) xs) (accuracy xs ts)

    check xs ts = return $
        let g = tsGeneration ts
            q = accuracy xs ts
        in  if g `mod` 100 == 0 && q >= 0.99
            then Just (g, q)
            else Nothing

-- Produce the result from the trained model and to compare the average mismatching ratio.
--
    accuracy xs ts = fromJust $ classifierAccuracyP' (tsModel ts) xs :: Double

data Iris = Setosa | Versicolor | Virginica deriving (Show, Read, Eq, Ord, Enum)

data Attributes = Attributes Double Double Double Double deriving (Show, Read, Eq, Ord)

type Sample = (Attributes, Iris)

-- From this parse it shows the discarding works because the implicit `f`
-- still do some job, due to the essence of Applicative. The only difference
-- between original sequence application is just that now only one result of application
-- get kept. For example, the result of parsing `char ','` is discarded because we 
-- don't need that.
sampleParser :: Parser Sample
sampleParser = f <$> (double <* char ',')
                 <*> (double <* char ',')
                 <*> (double <* char ',')
                 <*> (double <* char ',')
                 <*> irisParser
  where 
  
    f sl sw pl pw i = (Attributes sl sw pl pw, i)

    irisParser :: Parser Iris
    irisParser =     string "Iris-setosa"     *> return Setosa
                 <|> string "Iris-versicolor" *> return Versicolor
                 <|> string "Iris-virginica"  *> return Virginica

readSamples :: IO [Sample]
readSamples = do
    ls <- T.lines . T.pack <$> readFile ("examples" </> "iris" </> "data" <.> "csv")
    return $ f <$> ls

  where

    f l = let Right x = parseOnly sampleParser l in x

-- Classifier f n b c: classifies `b` into **categories** of type `c`,
-- f: input shape `f` (Vector 4 here)
-- n: output shape `Vector n`
type IrisModel = Classifier (Vector 4) 3 Attributes Iris

-- Layer 4 2: map from Vector length 4 (features: sepal len., sepal wid., petal len., pental wid.)
--                to Vector length 2 for Iris setosa and Iris virginica/versicolor (a mixed cluster)
--                Or, just a bool: True/False to "this" cluster, for single layer.
--
--                --> [True/False] --> Iris setosa
--                                 --> [True/False] --> Iris virginica
--                                                  --> Iris versicolor
--
-- The second argument just put the attributes into a `Vector 4 Double` vector.

irisModel :: IrisModel
irisModel = mkStdClassifier
    ((tanhLayer :: Layer 4 2) >>> tanhLayer)
    (\(Attributes sl sw pl pw) -> cons sl (cons sw (cons pl (cons pw nil)))) 
