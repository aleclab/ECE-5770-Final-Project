python copyImages.py --src-dir Images --dst-dir synthImages --per-image 200

python copyImages.py --src-dir Images --dst-dir synthImages --per-image 50 --choose-src random --seed 42


python generate_images_fast.py --count 80 --min_area 1e4 --max_area 1e10 --workers 2 --content solid


python generate_images_fast.py --count 80 --min_area 1e4 --max_area 1e10 --workers 1 --min_ar 1 --max_ar 1