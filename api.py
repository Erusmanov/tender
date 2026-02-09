import os
import json
import asyncio
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from parser import RostenderParser, Tender
from document import DocumentGenerator, load_company_profile, save_company_profile, _COMPANY_DEFAULTS

app = FastAPI(title="Tender Parser API", version="1.0.0")


def _safe_filepath(base_dir: str, *parts: str) -> str:
    """Resolve a file path and ensure it stays within base_dir. Raises HTTPException on traversal."""
    base = Path(base_dir).resolve()
    target = (base / Path(*parts)).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return str(target)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global parser instance (lazy init)
_parser: Optional[RostenderParser] = None
_generator: Optional[DocumentGenerator] = None

def get_parser() -> RostenderParser:
    global _parser
    if _parser is None:
        _parser = RostenderParser(
            email=os.getenv("ROSTENDER_EMAIL"),
            password=os.getenv("ROSTENDER_PASSWORD")
        )
    return _parser

def get_generator() -> DocumentGenerator:
    global _generator
    if _generator is None:
        _generator = DocumentGenerator()
    return _generator

# Models
class SearchRequest(BaseModel):
    region: str = ""
    industry: str = ""
    keyword: str

class TenderResponse(BaseModel):
    number: str
    title: str
    customer: str
    price: str
    deadline: str
    description: str
    url: str

class GenerateRequest(BaseModel):
    tender_url: str
    tender_number: str
    tender_title: str
    tender_customer: str
    tender_price: str
    tender_description: str

class GenerateResponse(BaseModel):
    proposal_docx: str
    proposal_pdf: str
    letter_docx: str
    letter_pdf: str
    requirements_docx: str
    requirements_pdf: str

# --- Company Profile model ---
class CompanyProfileRequest(BaseModel):
    name: str = ""
    full_name: str = ""
    details: str = ""
    address: str = ""
    contacts: str = ""
    director: str = ""
    experience: str = ""
    stack: str = ""
    advantages: str = ""

# Endpoints

@app.get("/api/company")
async def get_company_profile():
    """Get current company profile"""
    return {
        "profile": load_company_profile(),
        "fields": list(_COMPANY_DEFAULTS.keys()),
    }

@app.post("/api/company")
async def update_company_profile(request: CompanyProfileRequest):
    """Update company profile (saved to company.json)"""
    data = {k: v for k, v in request.model_dump().items() if v}
    if not data:
        raise HTTPException(status_code=400, detail="At least one field must be provided")

    # Merge with existing profile
    current = load_company_profile()
    current.update(data)

    if save_company_profile(current):
        return {"success": True, "profile": load_company_profile()}
    else:
        raise HTTPException(status_code=500, detail="Failed to save company profile")

@app.get("/api/regions")
async def get_regions():
    """Get list of regions"""
    try:
        with open("regions_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

@app.get("/api/branches")
async def get_branches():
    """Get list of industry branches"""
    try:
        with open("branches_data.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

@app.post("/api/search", response_model=List[TenderResponse])
async def search_tenders(request: SearchRequest):
    """Search for tenders"""
    if not request.keyword:
        raise HTTPException(status_code=400, detail="Keyword is required")
    
    parser = get_parser()
    tenders = await asyncio.to_thread(
        parser.search, request.region, request.industry, request.keyword
    )
    
    return [
        TenderResponse(
            number=t.number,
            title=t.title,
            customer=t.customer,
            price=t.price,
            deadline=t.deadline,
            description=t.description,
            url=t.url
        )
        for t in tenders
    ]

@app.get("/api/tender/details")
async def get_tender_details(url: str):
    """Get tender details including documents"""
    parser = get_parser()
    details = await asyncio.to_thread(parser.get_details, url)
    
    return {
        "description": details.get("description", ""),
        "requirements": details.get("requirements", ""),
        "documents": [
            {
                "id": doc.id,
                "title": doc.title,
                "url": doc.url,
                "extension": doc.extension
            }
            for doc in details.get("documents", [])
        ]
    }

class DownloadDocsRequest(BaseModel):
    tender_url: str
    tender_number: str

@app.post("/api/tender/download-docs")
async def download_tender_docs(request: DownloadDocsRequest):
    """Download all tender documents"""
    parser = get_parser()
    
    # Get details with document links
    details = await asyncio.to_thread(parser.get_details, request.tender_url)
    
    if not details.get("documents"):
        return {"success": False, "message": "Документы не найдены", "files": []}
    
    # Download files
    files_dir = await asyncio.to_thread(
        parser.download_files, details["documents"], request.tender_number
    )
    
    if not files_dir:
        return {"success": False, "message": "Ошибка скачивания", "files": []}
    
    # List downloaded files
    downloaded = []
    for f in os.listdir(files_dir):
        filepath = os.path.join(files_dir, f)
        if os.path.isfile(filepath):
            downloaded.append({
                "name": f,
                "path": filepath,
                "size": os.path.getsize(filepath),
                "download_url": f"/api/tender-files/{request.tender_number}/{f}"
            })
    
    return {
        "success": True, 
        "message": f"Скачано {len(downloaded)} файлов",
        "files": downloaded,
        "folder": files_dir
    }

@app.get("/api/tender-files/{tender_number}/{filename:path}")
async def get_tender_file(tender_number: str, filename: str):
    """Download a specific tender document"""
    filepath = _safe_filepath("downloads", tender_number, filename)
    return FileResponse(
        filepath,
        filename=Path(filename).name,
        media_type='application/octet-stream',
        headers={"Content-Disposition": f'attachment; filename="{Path(filename).name}"'}
    )

@app.get("/api/tender-files-zip/{tender_number}")
async def get_tender_files_zip(tender_number: str):
    """Download all tender documents as ZIP"""
    import zipfile
    import io

    folder = Path("downloads").resolve() / tender_number
    if not str(folder.resolve()).startswith(str(Path("downloads").resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, folder)
                zf.write(file_path, arc_name)
    
    zip_buffer.seek(0)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        zip_buffer,
        media_type='application/zip',
        headers={"Content-Disposition": f'attachment; filename="tender_{tender_number}_docs.zip"'}
    )

@app.get("/api/generated-zip/{tender_number}")
async def get_generated_zip(tender_number: str):
    """Download all generated documents as ZIP"""
    import zipfile
    import io

    folder = Path("outputs").resolve() / tender_number
    if not str(folder.resolve()).startswith(str(Path("outputs").resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                zf.write(file_path, file)
    
    zip_buffer.seek(0)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        zip_buffer,
        media_type='application/zip',
        headers={"Content-Disposition": f'attachment; filename="tender_{tender_number}_generated.zip"'}
    )

@app.post("/api/generate")
async def generate_documents(request: GenerateRequest):
    """Generate tender documents (proposal, letter, requirements)"""
    parser = get_parser()
    generator = get_generator()
    
    # Get full details (sync → thread)
    details = await asyncio.to_thread(parser.get_details, request.tender_url)
    
    # Download files if any (sync → thread)
    files_dir = ""
    if details.get("documents"):
        files_dir = await asyncio.to_thread(
            parser.download_files, details["documents"], request.tender_number
        )
    
    # Prepare data
    data = {
        "number": request.tender_number,
        "title": request.tender_title,
        "customer": request.tender_customer,
        "price": request.tender_price,
        "description": details.get("description") or request.tender_description,
        "requirements": details.get("requirements", "")
    }
    
    output_folder = f"outputs/{request.tender_number}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Generate documents (each OpenAI call is blocking → thread)
    results = {}
    
    # Proposal
    proposal_content = await asyncio.to_thread(generator.generate_proposal, data, files_dir)
    results["proposal_docx"] = generator.save_docx(proposal_content, f"{request.tender_number}_proposal", folder=output_folder)
    results["proposal_pdf"] = generator.save_pdf(proposal_content, f"{request.tender_number}_proposal", folder=output_folder) or ""
    
    # Letter
    letter_content = await asyncio.to_thread(generator.generate_letter, data, files_dir)
    results["letter_docx"] = generator.save_docx(letter_content, f"{request.tender_number}_letter", folder=output_folder)
    results["letter_pdf"] = generator.save_pdf(letter_content, f"{request.tender_number}_letter", folder=output_folder) or ""
    
    # Requirements
    req_content = await asyncio.to_thread(generator.generate_requirements, data, files_dir)
    results["requirements_docx"] = generator.save_docx(req_content, f"{request.tender_number}_requirements", folder=output_folder)
    results["requirements_pdf"] = generator.save_pdf(req_content, f"{request.tender_number}_requirements", folder=output_folder) or ""
    
    return results

@app.get("/api/download/{tender_number}/{filename}")
async def download_file(tender_number: str, filename: str):
    """Download generated document"""
    filepath = _safe_filepath("outputs", tender_number, filename)
    return FileResponse(
        filepath,
        filename=Path(filename).name,
        media_type='application/octet-stream',
        headers={"Content-Disposition": f'attachment; filename="{Path(filename).name}"'}
    )

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "api_key_set": bool(os.getenv("OPENAI_API_KEY"))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
