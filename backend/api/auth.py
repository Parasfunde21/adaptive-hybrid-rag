from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
)

from pydantic import BaseModel, Field

from services.auth_service import (
    authenticate_user,
    create_password_reset_token,
    create_session,
    create_user,
    create_verification_token,
    delete_session,
    get_user_from_token,
    reset_password,
    verify_email,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# Schemas
# ============================================================

class RegisterRequest(BaseModel):

    username: str = Field(
        min_length=3,
        max_length=30,
    )

    email: str = Field(
        min_length=3,
        max_length=100,
    )

    password: str = Field(
        min_length=6,
        max_length=100,
    )


class LoginRequest(BaseModel):

    username_or_email: str = Field(
        min_length=1,
        max_length=100,
    )

    password: str = Field(
        min_length=1,
        max_length=100,
    )


class ForgotPasswordRequest(BaseModel):

    email: str = Field(
        min_length=3,
        max_length=100,
    )


class ResetPasswordRequest(BaseModel):

    token: str = Field(
        min_length=10,
    )

    new_password: str = Field(
        min_length=6,
        max_length=100,
    )


# ============================================================
# Authentication Dependency
# ============================================================

def get_current_user(
    authorization: str | None = Header(default=None),
):

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    if not authorization.startswith(
        "Bearer "
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication header.",
        )

    token = authorization[
        len("Bearer "):
    ].strip()

    user = get_user_from_token(token)

    if user is None:

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session.",
        )

    return user


# ============================================================
# Register
# ============================================================

@router.post("/register")
def register(
    request: RegisterRequest,
):

    try:

        user = create_user(
            username=request.username,
            email=request.email,
            password=request.password,
        )

        verification_token = (
            create_verification_token(
                user["id"]
            )
        )

        token = create_session(
            user["id"]
        )

        return {
            "success": True,
            "message": (
                "Registration successful. "
                "Please verify your email."
            ),
            "user": user,
            "token": token,

            # Development mode.
            # Later this will be emailed.
            "verification_token":
                verification_token,
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# ============================================================
# Verify Email
# ============================================================

@router.get("/verify-email")
def verify_email_address(
    token: str,
):

    success = verify_email(
        token
    )

    if not success:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid or expired "
                "verification token."
            ),
        )

    return {
        "success": True,
        "message": "Email verified successfully.",
    }


# ============================================================
# Login
# ============================================================

@router.post("/login")
def login(
    request: LoginRequest,
):

    user = authenticate_user(
        username_or_email=
            request.username_or_email,
        password=request.password,
    )

    if user is None:

        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid username/email "
                "or password."
            ),
        )

    token = create_session(
        user["id"]
    )

    return {
        "success": True,
        "user": user,
        "token": token,
    }


# ============================================================
# Current User
# ============================================================

@router.get("/me")
def me(
    user=Depends(get_current_user),
):

    return {
        "success": True,
        "user": user,
    }


# ============================================================
# Forgot Password
# ============================================================

@router.post("/forgot-password")
def forgot_password(
    request: ForgotPasswordRequest,
):

    token = create_password_reset_token(
        request.email
    )

    # Do not reveal whether email exists.
    response = {
        "success": True,
        "message": (
            "If an account exists for this "
            "email, a password reset link "
            "has been generated."
        ),
    }

    # Development mode.
    # Later replace with email delivery.
    if token:

        response["reset_token"] = token

    return response


# ============================================================
# Reset Password
# ============================================================

@router.post("/reset-password")
def reset_password_endpoint(
    request: ResetPasswordRequest,
):

    try:

        success = reset_password(
            token=request.token,
            new_password=request.new_password,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    if not success:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid, expired, or already "
                "used reset token."
            ),
        )

    return {
        "success": True,
        "message": (
            "Password reset successfully. "
            "Please log in again."
        ),
    }


# ============================================================
# Logout
# ============================================================

@router.post("/logout")
def logout(
    authorization: str | None = Header(
        default=None
    ),
):

    if authorization and authorization.startswith(
        "Bearer "
    ):

        token = authorization[
            len("Bearer "):
        ].strip()

        delete_session(token)

    return {
        "success": True,
    }