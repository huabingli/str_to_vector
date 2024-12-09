from fastapi import APIRouter

from models.image_vector import ImageSimilarityBatch, ImageSimilarityOutBatch, ImageVector, ImageVectorOut
from schemas.base import R
from utils.openai import async_get_image_embedding, async_image_calculate_cosine_similarity

router = APIRouter(prefix='/image_vector', tags=['图片向量转换'])


@router.post(
        '/',
        summary='图片转向量',
        response_model=R[ImageVectorOut],
        response_model_exclude_none=True
)
async def get_image_vector(image_url: ImageVector):
    vector = await async_get_image_embedding(image_url.image_url)
    return R.success(data=ImageVectorOut(image_url=image_url.image_url, vector=vector))


@router.post('/image_similarity', summary='图片相似度计算', response_model=R[ImageSimilarityOutBatch])
async def get_image_similarity(images: ImageSimilarityBatch):
    data = await async_image_calculate_cosine_similarity(images)
    return R.success(data=data)
