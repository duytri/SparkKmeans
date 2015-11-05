package main.scala

import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.clustering.{ KMeans, KMeansModel }
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object KmeansDemo {
  def featurize(s: String): Vector = {
    val tf = new HashingTF(1000)
    val bigram = s.sliding(2).toSeq
    tf.transform(bigram)
  }

  def build_model(text: RDD[String], numClusters: Int, numIterations: Int): KMeansModel = {
    // Caches the vectors since it will be used many times by KMeans.
    val vectors = text.map(featurize).cache
    vectors.count() // Calls an action to create the cache.
    KMeans.train(vectors, numClusters, numIterations)
  }
  
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("KMeansDemo")
    val sc = new SparkContext(conf)
    val log = sc.textFile("/home/hduser/workspace/SparkKmeans/data/train.txt")
    val model = build_model(log, 10, 100)
    val predictedCluster = model.predict(featurize("My mother is very great"))
    println("-------------------------------------------------------")
    println(predictedCluster.toString())
    println("-------------------------------------------------------")
  }
}