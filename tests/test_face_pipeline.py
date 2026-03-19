import pytest
from httpx import AsyncClient, ASGITransport
from pathlib import Path
from PIL import Image
import io
import os
from app.main import app


@pytest.mark.asyncio
@pytest.mark.timeout(60)  # Таймаут 60 секунд на весь тест
async def test_enroll_and_verify():
    """Тест регистрации и верификации лица"""
    
    print("\n🔍 Начало теста enroll and verify...")
    
    IMAGE_PATH = Path(__file__).parent / "images" / "person1.jpg"
    
    # Проверяем размер файла
    file_size_mb = os.path.getsize(IMAGE_PATH) / (1024 * 1024)
    print(f"📁 Размер файла: {file_size_mb:.2f} MB")
    
    if file_size_mb > 5:
        print("⚠️  Файл слишком большой, выполняем ресайз...")
    
    # Оптимизируем изображение
    img = Image.open(IMAGE_PATH)
    print(f"📐 Размер изображения: {img.size}")
    
    # Ресайз если больше 1024px
    if max(img.size) > 1024:
        ratio = 1024 / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"📐 Новый размер: {img.size}")
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Конвертируем в bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=85, optimize=True)
    img_bytes.seek(0)
    
    final_size_kb = len(img_bytes.getvalue()) / 1024
    print(f"📦 Финальный размер: {final_size_kb:.1f} KB")
    
    async with AsyncClient(
        transport=ASGITransport(app=app), 
        base_url="http://test",
        timeout=30.0
    ) as client:
        
        # Enroll
        print("\n📝 Enroll...")
        img_bytes.seek(0)
        enroll = await client.post(
            "/upload?user_id=1",
            files={"file": ("person1.jpg", img_bytes, "image/jpeg")}
        )
        
        print(f"Status: {enroll.status_code}")
        assert enroll.status_code == 200
        print(f"✓ Enroll: {enroll.json()}")
        
        # Verify
        print("\n🔐 Verify...")
        img_bytes.seek(0)
        verify = await client.post(
            "/verify?user_id=1",
            files={"file": ("person1.jpg", img_bytes, "image/jpeg")}
        )
        
        print(f"Status: {verify.status_code}")
        assert verify.status_code == 200
        
        data = verify.json()
        print(f"✓ Verify: {data}")
        
        assert data["status"] == "match"
        assert data["similarity"] > 0.5
        
        print("\n✅ Тест завершен!")
