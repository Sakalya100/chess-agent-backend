from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from typing import List
import chess
import chess.svg
from autogen import ConversableAgent, register_function

version = "v1"
app = FastAPI(
    title="Chess Backend",
    description="A REST API Agentic Ches",
    version=version,
    docs_url=f"/api/{version}/docs",
    redoc_url=f"/api/{version}/redoc",
    contact={
        "email": "sakalyamitra@gmail.com"
    }
)

model_provider_map = {"OpenAI":"gpt-4o-mini", "Claude":"claude-3-5-sonnet-20240620", "Mixtral":"mistralai/Mixtral-8x7B-Instruct-v0.1", "Gemini":"gemini-2.0-flash"}

class Models(str, Enum):
    OPENAI = "OpenAI"
    CLAUDE = "Claude"
    MIXTRAL = "Mixtral"
    GEMINI = "Gemini"

class ChessRequest(BaseModel):
    white_model: Models
    black_model: Models
    white_api_key: str
    black_api_key: str
    max_turns: int = 5


@app.post("/play-chess")
def play_chess(req: ChessRequest):
    board = chess.Board()
    move_history: List[str] = []
    made_move = False

    def available_moves() -> str:
        return "Available moves are: " + ",".join(str(move) for move in board.legal_moves)

    def execute_move(move: str) -> str:
        nonlocal made_move
        try:
            chess_move = chess.Move.from_uci(move)
            if chess_move not in board.legal_moves:
                return f"Invalid move: {move}. Use available_moves() to check valid ones."

            board.push(chess_move)
            made_move = True

            move_history.append(move)

            moved_piece = board.piece_at(chess_move.to_square)
            piece_unicode = moved_piece.unicode_symbol()
            piece_name = chess.piece_name(moved_piece.piece_type).capitalize()
            from_sq = chess.SQUARE_NAMES[chess_move.from_square]
            to_sq = chess.SQUARE_NAMES[chess_move.to_square]

            move_desc = f"Moved {piece_name} ({piece_unicode}) from {from_sq} to {to_sq}."

            if board.is_checkmate():
                winner = 'White' if board.turn == chess.BLACK else 'Black'
                move_desc += f"\nCheckmate! {winner} wins!"
            elif board.is_stalemate():
                move_desc += "\nStalemate!"
            elif board.is_insufficient_material():
                move_desc += "\nDraw due to insufficient material."
            elif board.is_check():
                move_desc += "\nCheck!"

            return move_desc
        except Exception as e:
            return f"Error in move {move}: {str(e)}"

    def check_made_move(msg):
        nonlocal made_move
        if made_move:
            made_move = False
            return True
        return False

    white_model = model_provider_map[req.white_model]
    black_model = model_provider_map[req.black_model]
    agent_white_config = [{"model": white_model, "api_key": req.white_api_key}]
    agent_black_config = [{"model": black_model, "api_key": req.black_api_key}]

    agent_white = ConversableAgent(
        name="Agent_White",
        system_message="You are a professional chess player and you play as white. "
                       "First call available_moves() to get legal moves, then call execute_move(move) to play.",
        llm_config={"config_list": agent_white_config, "cache_seed": None}
    )

    agent_black = ConversableAgent(
        name="Agent_Black",
        system_message="You are a professional chess player and you play as black. "
                       "First call available_moves() to get legal moves, then call execute_move(move) to play.",
        llm_config={"config_list": agent_black_config, "cache_seed": None}
    )

    game_master = ConversableAgent(
        name="Game_Master",
        llm_config=False,
        is_termination_msg=check_made_move,
        default_auto_reply="Please make a move.",
        human_input_mode="NEVER"
    )

    for agent in [agent_white, agent_black]:
        register_function(execute_move, caller=agent, executor=game_master, name="execute_move", description="Make a move.")
        register_function(available_moves, caller=agent, executor=game_master, name="available_moves", description="List legal moves.")

    agent_white.register_nested_chats(
        trigger=agent_black,
        chat_queue=[{"sender": game_master, "recipient": agent_white, "summary_method": "last_msg"}]
    )

    agent_black.register_nested_chats(
        trigger=agent_white,
        chat_queue=[{"sender": game_master, "recipient": agent_black, "summary_method": "last_msg"}]
    )

    chat_result = agent_black.initiate_chat(
        recipient=agent_white,
        message="Let's play chess! You go first.",
        max_turns=req.max_turns,
        summary_method="reflection_with_llm"
    )

    return {"move_history": move_history, "summary": chat_result.summary, "models": {"white": white_model, "black": black_model}}
