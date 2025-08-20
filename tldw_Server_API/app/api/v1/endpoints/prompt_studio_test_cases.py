# prompt_studio_test_cases.py
# API endpoints for test case management in Prompt Studio

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from loguru import logger

# Local imports
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import StandardResponse, ListResponse
from tldw_Server_API.app.api.v1.schemas.prompt_studio_test import (
    TestCaseCreate, TestCaseUpdate, TestCaseResponse, TestCaseBulkCreate,
    TestCaseImportRequest, TestCaseExportRequest, TestCaseGenerateRequest
)
from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_db, get_prompt_studio_user, require_project_write_access,
    check_rate_limit, get_security_config, PromptStudioDatabase, SecurityConfig
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_manager import TestCaseManager
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_io import TestCaseIO
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_case_generator import TestCaseGenerator
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import DatabaseError, InputError

########################################################################################################################
# Router Setup

router = APIRouter(
    prefix="/api/v1/prompt_studio/test_cases",
    tags=["Prompt Studio - Test Cases"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"}
    }
)

########################################################################################################################
# Test Case CRUD Endpoints

@router.post("/create", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
async def create_test_case(
    test_case_data: TestCaseCreate,
    _: bool = Depends(lambda: require_project_write_access(test_case_data.project_id)),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Create a new test case.
    
    Args:
        test_case_data: Test case creation data
        db: Database instance
        security_config: Security configuration
        user_context: Current user context
        
    Returns:
        Created test case details
    """
    try:
        # Check test case limit
        manager = TestCaseManager(db)
        current_count = manager.get_test_case_stats(test_case_data.project_id)["total"]
        
        if current_count >= security_config.max_test_cases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Project has reached maximum of {security_config.max_test_cases} test cases"
            )
        
        # Create test case
        test_case = manager.create_test_case(
            project_id=test_case_data.project_id,
            name=test_case_data.name,
            description=test_case_data.description,
            inputs=test_case_data.inputs,
            expected_outputs=test_case_data.expected_outputs,
            tags=test_case_data.tags,
            is_golden=test_case_data.is_golden,
            signature_id=test_case_data.signature_id
        )
        
        logger.info(f"User {user_context['user_id']} created test case: {test_case.get('name', 'Unnamed')}")
        
        return StandardResponse(
            success=True,
            data=TestCaseResponse(**test_case)
        )
        
    except DatabaseError as e:
        logger.error(f"Database error creating test case: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create test case"
        )

@router.post("/bulk", response_model=StandardResponse, status_code=status.HTTP_201_CREATED)
async def create_bulk_test_cases(
    bulk_data: TestCaseBulkCreate,
    _: bool = Depends(lambda: require_project_write_access(bulk_data.project_id)),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Create multiple test cases at once.
    
    Args:
        bulk_data: Bulk test case creation data
        db: Database instance
        security_config: Security configuration
        user_context: Current user context
        
    Returns:
        List of created test cases
    """
    try:
        manager = TestCaseManager(db)
        
        # Check test case limit
        current_count = manager.get_test_case_stats(bulk_data.project_id)["total"]
        new_total = current_count + len(bulk_data.test_cases)
        
        if new_total > security_config.max_test_cases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Would exceed maximum of {security_config.max_test_cases} test cases"
            )
        
        # Create test cases
        test_cases = manager.create_bulk_test_cases(
            project_id=bulk_data.project_id,
            test_cases=[tc.dict() for tc in bulk_data.test_cases],
            signature_id=bulk_data.signature_id
        )
        
        logger.info(f"User {user_context['user_id']} created {len(test_cases)} test cases in bulk")
        
        return StandardResponse(
            success=True,
            data=[TestCaseResponse(**tc) for tc in test_cases]
        )
        
    except DatabaseError as e:
        logger.error(f"Database error creating bulk test cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create test cases"
        )

@router.get("/list/{project_id}", response_model=ListResponse)
async def list_test_cases(
    project_id: int = Path(..., description="Project ID"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    is_golden: Optional[bool] = Query(None, description="Filter by golden status"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    signature_id: Optional[int] = Query(None, description="Filter by signature"),
    _: bool = Depends(lambda: require_project_access(project_id)),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> ListResponse:
    """
    List test cases in a project.
    
    Args:
        project_id: Project ID
        page: Page number
        per_page: Items per page
        is_golden: Filter by golden status
        tags: Filter by tags
        search: Search query
        signature_id: Filter by signature
        db: Database instance
        
    Returns:
        Paginated list of test cases
    """
    try:
        manager = TestCaseManager(db)
        
        # Parse tags
        tag_list = None
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        
        # Get test cases
        result = manager.list_test_cases(
            project_id=project_id,
            signature_id=signature_id,
            is_golden=is_golden,
            tags=tag_list,
            search=search,
            page=page,
            per_page=per_page
        )
        
        return ListResponse(
            success=True,
            data=[TestCaseResponse(**tc) for tc in result["test_cases"]],
            metadata=result["pagination"]
        )
        
    except DatabaseError as e:
        logger.error(f"Database error listing test cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list test cases"
        )

@router.get("/get/{test_case_id}", response_model=StandardResponse)
async def get_test_case(
    test_case_id: int = Path(..., description="Test case ID"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> StandardResponse:
    """
    Get a specific test case by ID.
    
    Args:
        test_case_id: Test case ID
        db: Database instance
        
    Returns:
        Test case details
    """
    try:
        manager = TestCaseManager(db)
        test_case = manager.get_test_case(test_case_id)
        
        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test case {test_case_id} not found"
            )
        
        # Check project access
        await require_project_access(test_case["project_id"])
        
        return StandardResponse(
            success=True,
            data=TestCaseResponse(**test_case)
        )
        
    except DatabaseError as e:
        logger.error(f"Database error getting test case: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get test case"
        )

@router.put("/update/{test_case_id}", response_model=StandardResponse)
async def update_test_case(
    test_case_id: int = Path(..., description="Test case ID"),
    updates: TestCaseUpdate = ...,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Update a test case.
    
    Args:
        test_case_id: Test case ID
        updates: Fields to update
        db: Database instance
        user_context: Current user context
        
    Returns:
        Updated test case details
    """
    try:
        manager = TestCaseManager(db)
        
        # Get test case to check project
        test_case = manager.get_test_case(test_case_id)
        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test case {test_case_id} not found"
            )
        
        # Check project access
        await require_project_write_access(test_case["project_id"])
        
        # Update test case
        update_data = {k: v for k, v in updates.dict().items() if v is not None}
        updated = manager.update_test_case(test_case_id, update_data)
        
        logger.info(f"User {user_context['user_id']} updated test case {test_case_id}")
        
        return StandardResponse(
            success=True,
            data=TestCaseResponse(**updated)
        )
        
    except InputError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DatabaseError as e:
        logger.error(f"Database error updating test case: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update test case"
        )

@router.delete("/delete/{test_case_id}", response_model=StandardResponse)
async def delete_test_case(
    test_case_id: int = Path(..., description="Test case ID"),
    permanent: bool = Query(False, description="Permanently delete"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Delete a test case.
    
    Args:
        test_case_id: Test case ID
        permanent: If True, permanently delete
        db: Database instance
        user_context: Current user context
        
    Returns:
        Success response
    """
    try:
        manager = TestCaseManager(db)
        
        # Get test case to check project
        test_case = manager.get_test_case(test_case_id)
        if not test_case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test case {test_case_id} not found"
            )
        
        # Check project access
        await require_project_write_access(test_case["project_id"])
        
        # Delete test case
        success = manager.delete_test_case(test_case_id, hard_delete=permanent)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test case not found or already deleted"
            )
        
        logger.info(
            f"User {user_context['user_id']} {'permanently' if permanent else 'soft'} "
            f"deleted test case {test_case_id}"
        )
        
        return StandardResponse(
            success=True,
            data={"message": f"Test case {'permanently' if permanent else 'soft'} deleted"}
        )
        
    except DatabaseError as e:
        logger.error(f"Database error deleting test case: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete test case"
        )

########################################################################################################################
# Import/Export Endpoints

@router.post("/import", response_model=StandardResponse)
async def import_test_cases(
    import_data: TestCaseImportRequest,
    _: bool = Depends(lambda: require_project_write_access(import_data.project_id)),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Import test cases from CSV or JSON.
    
    Args:
        import_data: Import request data
        db: Database instance
        security_config: Security configuration
        user_context: Current user context
        
    Returns:
        Import results
    """
    try:
        manager = TestCaseManager(db)
        io_manager = TestCaseIO(manager)
        
        # Check test case limit
        current_count = manager.get_test_case_stats(import_data.project_id)["total"]
        
        # Import based on format
        if import_data.format == "csv":
            imported, errors = io_manager.import_from_csv(
                project_id=import_data.project_id,
                csv_data=import_data.data,
                signature_id=import_data.signature_id,
                auto_generate_names=import_data.auto_generate_names
            )
        else:  # json
            imported, errors = io_manager.import_from_json(
                project_id=import_data.project_id,
                json_data=import_data.data,
                signature_id=import_data.signature_id,
                auto_generate_names=import_data.auto_generate_names
            )
        
        # Check if we exceeded the limit
        new_total = current_count + imported
        if new_total > security_config.max_test_cases:
            logger.warning(f"Import would exceed test case limit for project {import_data.project_id}")
        
        logger.info(f"User {user_context['user_id']} imported {imported} test cases")
        
        return StandardResponse(
            success=True,
            data={
                "imported": imported,
                "errors": errors,
                "total_test_cases": new_total
            }
        )
        
    except Exception as e:
        logger.error(f"Error importing test cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to import test cases"
        )

@router.post("/export/{project_id}", response_model=StandardResponse)
async def export_test_cases(
    project_id: int = Path(..., description="Project ID"),
    export_request: TestCaseExportRequest = ...,
    _: bool = Depends(lambda: require_project_access(project_id)),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
) -> StandardResponse:
    """
    Export test cases to CSV or JSON.
    
    Args:
        project_id: Project ID
        export_request: Export configuration
        db: Database instance
        
    Returns:
        Exported data
    """
    try:
        manager = TestCaseManager(db)
        io_manager = TestCaseIO(manager)
        
        # Export based on format
        if export_request.format == "csv":
            data = io_manager.export_to_csv(
                project_id=project_id,
                include_golden_only=export_request.include_golden_only,
                tag_filter=export_request.tag_filter
            )
        else:  # json
            data = io_manager.export_to_json(
                project_id=project_id,
                include_golden_only=export_request.include_golden_only,
                tag_filter=export_request.tag_filter
            )
        
        return StandardResponse(
            success=True,
            data={
                "format": export_request.format,
                "data": data,
                "content_type": "text/csv" if export_request.format == "csv" else "application/json"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting test cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export test cases"
        )

########################################################################################################################
# Generation Endpoints

@router.post("/generate", response_model=StandardResponse)
async def generate_test_cases(
    generate_request: TestCaseGenerateRequest,
    _: bool = Depends(lambda: require_project_write_access(generate_request.project_id)),
    _rate: bool = Depends(lambda: check_rate_limit("generate")),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    security_config: SecurityConfig = Depends(get_security_config),
    user_context: Dict = Depends(get_prompt_studio_user)
) -> StandardResponse:
    """
    Auto-generate test cases.
    
    Args:
        generate_request: Generation configuration
        db: Database instance
        security_config: Security configuration
        user_context: Current user context
        
    Returns:
        Generated test cases
    """
    try:
        manager = TestCaseManager(db)
        generator = TestCaseGenerator(manager)
        
        # Check test case limit
        current_count = manager.get_test_case_stats(generate_request.project_id)["total"]
        new_total = current_count + generate_request.num_cases
        
        if new_total > security_config.max_test_cases:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Would exceed maximum of {security_config.max_test_cases} test cases"
            )
        
        # Generate based on strategy
        if generate_request.generation_strategy == "diverse" and generate_request.signature_id:
            generated = generator.generate_diverse_cases(
                project_id=generate_request.project_id,
                signature_id=generate_request.signature_id,
                num_cases=generate_request.num_cases
            )
        elif generate_request.base_on_description:
            generated = generator.generate_from_description(
                project_id=generate_request.project_id,
                description=generate_request.base_on_description,
                num_cases=generate_request.num_cases,
                signature_id=generate_request.signature_id,
                prompt_id=generate_request.prompt_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide either signature_id for diverse generation or base_on_description"
            )
        
        logger.info(f"User {user_context['user_id']} generated {len(generated)} test cases")
        
        return StandardResponse(
            success=True,
            data=[TestCaseResponse(**tc) for tc in generated]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating test cases: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate test cases"
        )