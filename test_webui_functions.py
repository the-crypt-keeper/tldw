#!/usr/bin/env python3
"""
Playwright test to verify WebUI chat functions are properly defined and working.
"""

import asyncio
import time
from playwright.async_api import async_playwright, expect


async def test_chat_functions():
    """Test that chat functions are properly defined in the WebUI."""
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)  # Set headless=True for CI
        context = await browser.new_context()
        page = await context.new_page()
        
        # Enable console logging to capture JavaScript errors
        page.on("console", lambda msg: print(f"Browser console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"Browser error: {err}"))
        
        try:
            print("1. Navigating to WebUI...")
            await page.goto("http://127.0.0.1:8000/webui/")
            await page.wait_for_load_state("networkidle")
            
            # Wait for initial page load
            await page.wait_for_timeout(2000)
            
            print("2. Clicking on Chat tab...")
            # Click on Chat top-level tab
            await page.click('button[data-toptab="chat"]')
            await page.wait_for_timeout(1000)
            
            # Wait for sub-tabs to appear and click on Chat Completions
            print("   Waiting for Chat Completions sub-tab...")
            # The sub-tab might already be active or we need to wait for it
            try:
                # Try to click the sub-tab if it exists and is visible
                sub_tab = await page.wait_for_selector('button[data-subtab="tabChatCompletions"]', timeout=5000)
                if sub_tab:
                    await sub_tab.click()
                    print("   Clicked Chat Completions sub-tab")
            except:
                # Sub-tab might already be active or loaded
                print("   Chat Completions tab may already be active")
            
            await page.wait_for_timeout(1000)
            
            print("3. Checking if functions are defined...")
            # Check if our functions are defined in the global scope
            functions_check = await page.evaluate("""
                () => {
                    const functions = [
                        'makeChatCompletionsRequest',
                        'sendChatMessage',
                        'clearChat',
                        'toggleLogprobs',
                        'toggleToolChoiceJSON',
                        'exportCharacter',
                        'createCharacter',
                        'listCharacters'
                    ];
                    
                    const results = {};
                    functions.forEach(fn => {
                        results[fn] = {
                            defined: typeof window[fn] !== 'undefined',
                            type: typeof window[fn]
                        };
                    });
                    return results;
                }
            """)
            
            print("\nFunction availability check:")
            all_defined = True
            for func_name, info in functions_check.items():
                status = "✓" if info['defined'] else "✗"
                print(f"  {status} {func_name}: {info['type']}")
                if not info['defined']:
                    all_defined = False
            
            if not all_defined:
                print("\n⚠️  Some functions are not defined!")
                
                # Try to check if tab-functions.js was loaded
                script_loaded = await page.evaluate("""
                    () => {
                        const scripts = Array.from(document.querySelectorAll('script'));
                        return scripts.some(s => s.src.includes('tab-functions.js'));
                    }
                """)
                print(f"  tab-functions.js in page: {script_loaded}")
            else:
                print("\n✅ All functions are properly defined!")
            
            print("\n4. Testing makeChatCompletionsRequest function...")
            # Try to call the function (it will fail without proper API setup, but we can check if it exists)
            try:
                # First, set up a minimal valid message
                await page.fill('#chatCompletions_messages', '[{"role": "user", "content": "test"}]')
                
                # Try to click the send button
                button_exists = await page.is_visible('button:has-text("Send Request")')
                print(f"  Send Request button visible: {button_exists}")
                
                if button_exists:
                    # Check if clicking the button triggers the function
                    error_occurred = await page.evaluate("""
                        async () => {
                            try {
                                // Check if function exists
                                if (typeof makeChatCompletionsRequest === 'undefined') {
                                    return 'Function not defined';
                                }
                                // Try to call it (will fail on API call, but proves function exists)
                                await makeChatCompletionsRequest();
                                return 'Function called successfully';
                            } catch (e) {
                                return `Function called but got error: ${e.message}`;
                            }
                        }
                    """)
                    print(f"  Function test result: {error_occurred}")
            except Exception as e:
                print(f"  Error testing function: {e}")
            
            print("\n5. Testing interactive chat interface...")
            # Check if chat interface elements exist
            chat_input_exists = await page.is_visible('#chat-input')
            send_button_exists = await page.is_visible('.btn.btn-primary:has-text("Send")')
            clear_button_exists = await page.is_visible('button:has-text("Clear Chat")')
            
            print(f"  Chat input field exists: {chat_input_exists}")
            print(f"  Send button exists: {send_button_exists}")
            print(f"  Clear Chat button exists: {clear_button_exists}")
            
            if all([chat_input_exists, send_button_exists, clear_button_exists]):
                print("  ✅ Interactive chat interface elements found")
            else:
                print("  ⚠️  Some chat interface elements missing")
            
            print("\n6. Final verification...")
            # Final check - try to execute a simple function
            result = await page.evaluate("""
                () => {
                    if (typeof clearChat === 'function') {
                        // Function exists, we can call it
                        try {
                            clearChat();
                            return 'clearChat executed successfully';
                        } catch (e) {
                            return `clearChat exists but failed: ${e.message}`;
                        }
                    }
                    return 'clearChat function not found';
                }
            """)
            print(f"  {result}")
            
            # Return test results
            return all_defined
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            await browser.close()


async def main():
    """Main test runner."""
    print("=" * 60)
    print("WebUI Chat Functions Test")
    print("=" * 60)
    
    # Run the test
    success = await test_chat_functions()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: All functions are properly defined")
    else:
        print("❌ TEST FAILED: Some functions are not available")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)