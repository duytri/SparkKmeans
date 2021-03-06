name := "Spark Demo Applications"
version := "1.0"
organization := "uit.islab"
scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.1",
  "org.apache.spark" %% "spark-mllib" % "1.5.1" % "provided",
  "org.apache.spark" %% "spark-sql" % "1.5.1" % "provided",
  "org.apache.spark" % "spark-streaming_2.10" % "1.4.1" % "provided",
  "org.apache.spark" % "spark-streaming-kafka_2.10" % "1.4.1"
)
