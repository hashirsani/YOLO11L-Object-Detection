 1):-  How long did your first working version take on the full 50 files, and how long did your final
version take? Which single change saved the most time?

MY first version using fp16 and 4 sample per-second ran in 12,720 seconds before running on full data 
I tested on a small dataset of 14 videos and on different setting using FP32 and FP16 and sample sizes 
of 2,3,4,5,6,7,8 i found out that FP16 is 2 times faster than FP32 and checking 4 frames persecond is 
superior in time consumption and all other matrics 

 2):-  Describe what your script does, step by step, from opening a Parquet file to writing an
output line.

My script opens each parquet file and read rows from mp4 then uses av to check the rotation of frames 
if not verical it records th timestamp and fixes it then takes 4 frames per second convert the picters in BGR
and runs yolo11l.pt at imgsz=640, conf=0.4, keeping only person, car, dog, laptop, and cell phone.Marks an object 
present if seen in at least 0.5 OBJ_FRAC of sampled frames in that second, and takes the median of person counts.
Writes one JSON line per second (video, second, objects, people) and catches per-video failures so already-written 
lines stay intact and one error line is emitted for a broken video.

3):-  Which part of your final version takes the most time? How did you measure this?

Shard download was 101s, while the 50 shards took 12,720s (half1 = 6,509s, half2 = 6,211s,
 averaging ~254s per shard) and final gzip/validation took ~3.7s (12,850.0s to 12,853.7s) it
is measured by kaggle logs timestamp per shard.

4):- How did you check that your results stayed correct after you made the script
faster?


It is checked by object micro F1 score people exact match and people within 1

5):-  Describe one suggestion from an AI tool that was wrong or that you decided not to use.
How did you find out it was wrong, or why did you reject it?

   The sugession of using frames 2 per second I rejected it because by using 2 frames there was some trade off
the micro F1 score was 0.9627 but in 4 frames it was 0.9811 and people exact match from 0.95 to 0.92 the time of 
4 frames per second was well under time limit

6):-  If you had one more week, what would you change, and why?

I would try a new approach in tht approach i will use 2 frames per second and use more frames only when those 2 frames disagree with each other this 
wil save a lot of time as compared to using strict 4 frames and also the F1 will be greater than using the 2 frames per second